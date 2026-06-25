import torch
from torch import nn
from torch.nn.functional import gelu


class TwoDimensionalAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.cross_column_layer = TransformerLayer(config)
        self.cross_row_layer = TransformerLayer(config)

    def forward_horizontal(
        self, hidden_states: torch.Tensor, num_features: int, kv: torch.Tensor = None, attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Apply horizontal (cross-column) attention to the hidden states.
        We simply treat the rows as batch elements.
        attn_mask: optional (batch_size, num_feature_cols) bool mask, True = attend, False = skip (padding).
        """
        batch_size, num_rows, num_columns, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * num_rows, num_columns, hidden_size)
        current_batch_size = batch_size * num_rows
        max_rows_per_batch = 8192
        col_fraction = 100.0 / float(num_columns)
        batch_step = max(1, int(max_rows_per_batch * col_fraction))

        kv_flat = None
        if kv is not None:
            kv_flat = kv.view(batch_size * num_rows, kv.shape[2], kv.shape[3])

        # Expand attn_mask from (batch_size, num_feature_cols) to (batch_size * num_rows, 1, 1, num_feature_cols)
        attn_mask_flat = None
        if attn_mask is not None:
            if attn_mask.ndim == 2:  # (B, C) — same mask for all rows
                attn_mask_flat = (
                    attn_mask.unsqueeze(1).expand(-1, num_rows, -1).reshape(batch_size * num_rows, 1, 1, -1)
                )
            else:  # (B, R, C) — per-row mask
                attn_mask_flat = attn_mask.reshape(batch_size * num_rows, 1, 1, -1)

        if current_batch_size <= batch_step:
            # Always in training and often in inference, typical R <= 1000, C <= 100 ---> 1 chunk. Use fast path.
            horizontal_outputs = self.cross_column_layer(
                hidden_states, num_features, kv=kv_flat, attn_mask=attn_mask_flat
            )
        else:
            horizontal_outputs = torch.empty_like(hidden_states)

            for i in range(0, current_batch_size, batch_step):
                end_idx = i + batch_step
                chunk = hidden_states[i:end_idx, :, :]
                if kv_flat is not None:
                    chunk_output = self.cross_column_layer(chunk, num_features, kv=kv_flat[i:end_idx])
                else:
                    chunk_output = self.cross_column_layer(chunk, num_features)
                horizontal_outputs[i : i + batch_step] = chunk_output

        return horizontal_outputs.view(batch_size, num_rows, num_columns, hidden_size)

    def forward_vertical(self, horizontal_outputs: torch.Tensor, num_contexts: int) -> torch.Tensor:
        """horizontal_outputs: shape (batch_size, num_rows, num_columns, hidden_size)
        num_contexts: how many elements (i.e. context rows) to attend to
        Apply vertical (cross-row) attention to the hidden states.
        We need permute the hidden states to treat the columns as batch elements.
        """
        # For cross-row attention we can do the same, but we need to permute before and after.
        # horizontal_outputs has shape (batch_size, num_rows, num_columns, hidden_size)
        batch_step = 100
        batch_size, num_rows, num_columns, hidden_size = horizontal_outputs.shape
        horizontal_outputs = (
            horizontal_outputs.transpose(1, 2).contiguous().view(batch_size * num_columns, num_rows, hidden_size)
        )

        total = horizontal_outputs.shape[0]
        if total <= batch_step:
            # Always in training and often in inference, typical R <= 1000, C <= 100 => 1 chunk. Use fast path.
            vertical_outputs = self.cross_row_layer(horizontal_outputs, num_contexts)
        else:
            vertical_outputs = torch.empty_like(horizontal_outputs)
            for i in range(0, total, batch_step):
                chunk = horizontal_outputs[i : i + batch_step]
                chunk_output = self.cross_row_layer(chunk, num_contexts)
                vertical_outputs[i : i + batch_step] = chunk_output
        return vertical_outputs.view(batch_size, num_columns, num_rows, hidden_size).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_contexts: int,
        num_features: int = None,
        kv: torch.Tensor = None,
        columns_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """hidden_states: shape (batch_size, num_rows, num_columns, hidden_size)
        num_contexts: how many elements (i.e. context rows) to attend to
        num_features: how many elements (i.e. feature columns, or all columns for "full" attention) to attend to
        columns_attn_mask: optional (batch_size, num_feature_cols) bool mask for horizontal attention padding.
        Applies the two separate TransformerLayers, once along the "columns" direction and once along the "rows" direction.

        Returns: tensor of shape (batch_size, num_rows, num_columns, hidden_size)
        """
        hidden_states = self.forward_horizontal(hidden_states, num_features, kv=kv, attn_mask=columns_attn_mask)

        return self.forward_vertical(hidden_states, num_contexts)


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TorchAttention(config)
        self.intermediate = FeedForwardIntermediate(config)
        self.output = FeedForwardOutput(config)

    def forward(
        self, hidden_states: torch.Tensor, num_to_attend: int, kv: torch.Tensor = None, attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states, num_to_attend, kv=kv, attn_mask=attn_mask)
        intermediate_output = self.intermediate(attention_output)
        return self.output(intermediate_output, attention_output)


class FeedForwardIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class FeedForwardOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AttentionOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TorchAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = TorchSelfAttention(config)
        self.output = AttentionOutput(config)

    def forward(
        self, hidden_states: torch.Tensor, num_to_attend: int, kv: torch.Tensor = None, attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        self_outputs = self.self_attention(hidden_states, num_to_attend, kv=kv, attn_mask=attn_mask)
        return self.output(self_outputs, hidden_states)


class TorchSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(
                f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention '
                f'heads ({config.num_attention_heads})'
            )

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0, (
            f'{config.hidden_size=} must be divisible by {config.num_attention_heads=}'
        )
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.kv_proj = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def forward(
        self, hidden_states: torch.Tensor, num_to_attend: int, kv: torch.Tensor = None, attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """hidden_states: shape (batch_size, seq_len, hidden_size)
        num_to_attend: how many elements from hidden_states to attend to.
        attn_mask: optional bool mask (batch_size, 1, 1, num_kv), True = attend, False = skip.
        """
        query = self.transpose_for_scores(self.q_proj(hidden_states))

        if kv is not None:
            kv_source = kv
            kv_fused = self.kv_proj(kv)
            bool_mask = ~(kv_source == 0).all(dim=-1)  # (batch_size, kv_seq_len)
            attn_mask = bool_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, kv_seq_len)
        else:
            kv_fused = self.kv_proj(hidden_states[:, :num_to_attend, :])
        key, value = torch.chunk(kv_fused, 2, dim=-1)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # TODO: when using bagging, it's probably better to leave dropout on also in eval mode
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )

        context_layer = attn_output.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.hidden_size,)
        return context_layer.view(new_context_layer_shape)
