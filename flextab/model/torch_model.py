import os
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from flextab.constants import MAX_CLASSIFICATION_CLASSES, NUM_VALID_BINS, TaskType
from flextab.model.attention import TwoDimensionalAttentionLayer
from flextab.model.config import FlexTabConfig
from flextab.model.embeddings import CellEmbeddings

os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'


class RoundUpTo8LinearLayer(nn.Linear):
    """Functionally equivalent to nn.Linear with same parameters.
    However, apparently deepspeed has issues with nn.Linear of size not multiple of 8,
    so we use this class instead to round up the output size to a multiple of 8.
    """

    def __init__(self, in_features: int, out_features: int):
        rounded_out_features = ceil(out_features / 8) * 8
        self.real_out_features = out_features
        super().__init__(in_features, rounded_out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = super().forward(input)
        return result[..., : self.real_out_features]


class Head(nn.Module):
    def __init__(self, config: FlexTabConfig):
        super().__init__()
        self.config = config

        # standard class prediction head
        self.dense_classif = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        # should be just +1, but for backward compatibility we have +2
        self.output_head_classif = nn.Linear(self.config.hidden_size, MAX_CLASSIFICATION_CLASSES + 2)

        self.dense_reg = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.output_head_reg = RoundUpTo8LinearLayer(self.config.hidden_size, NUM_VALID_BINS)

    def forward(self, encoder_outputs: torch.Tensor, is_regression: bool):
        """Last part of the "forward" method.
        It takes the encoder outputs (one token per row) and applies the heads
        """
        is_classification = not is_regression

        if is_classification:
            out = self.dense_classif(encoder_outputs)
            out = F.gelu(out)
            out = self.output_head_classif(out)
        else:
            out = self.dense_reg(encoder_outputs)
            out = F.gelu(out)
            out = self.output_head_reg(out)

        return out


class FlexTab(nn.Module):
    def __init__(
        self,
        num_decoder_layers: int = 12,
        num_encoder_layers: int = 12,
        hidden_size: int = 768,
        num_row_tokens: int = 1,
        bottleneck_size: int = 768,
        **kwargs,
    ):
        super().__init__()

        self.num_row_tokens = num_row_tokens

        self.config = FlexTabConfig(
            num_hidden_layers=num_encoder_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=hidden_size // 64,
            bottleneck_size=bottleneck_size,
            layer_norm_eps=1e-5,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

        self.embeddings = CellEmbeddings(self.config)
        self.in_context_encoder = nn.ModuleList(
            [TwoDimensionalAttentionLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )

        self.in_context_decoder = nn.ModuleList(
            [TwoDimensionalAttentionLayer(self.config) for _ in range(num_decoder_layers)]
        )

        self.heads = self.construct_heads()

        self.compression_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.config.hidden_size, self.config.bottleneck_size)
                        for _ in range(self.num_row_tokens)
                    ]
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )

        self.up_proj = nn.Linear(self.config.bottleneck_size, self.config.hidden_size)
        self.row_token = nn.Parameter(torch.zeros((1, 1, self.num_row_tokens, self.config.hidden_size)))

    def construct_heads(self) -> nn.Module:
        return Head(self.config)

    def load_weights(self, checkpoint_path_or_state_dict: str | Path | dict, device: torch.device):
        if isinstance(checkpoint_path_or_state_dict, (str, Path)):
            state_dict = torch.load(checkpoint_path_or_state_dict, map_location=device, weights_only=True)
        else:
            state_dict = checkpoint_path_or_state_dict
        if all([k.startswith('module.') for k in state_dict]):
            # Remove module. in front of all keys - seems to be added by deepspeed by default
            state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict)

    def extract_prediction_classification(self, logits: torch.Tensor, label_classes: np.ndarray):
        logits = logits[:, : len(label_classes[0])].cpu().float()
        preds_indices = torch.argmax(logits, dim=-1).numpy()

        if label_classes.shape[0] == 1:
            preds = label_classes[0][preds_indices]
        else:
            preds = label_classes[np.arange(len(preds_indices)), preds_indices]
        return preds, logits

    def extract_prediction_regression(
        self,
        logits: torch.Tensor,
        label_classes: np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(label_classes, np.ndarray):
            label_classes = torch.tensor(label_classes, dtype=torch.float32)

        probas = torch.softmax(logits[:, : len(label_classes[0])], dim=1).cpu().float()
        preds = torch.sum(probas * label_classes, dim=1)
        return preds, probas

    def forward_encoder(self, feature_embeddings: torch.Tensor, num_context_rows: int) -> torch.Tensor:
        batch_size, num_rows, _, _ = feature_embeddings.shape
        current_row_tokens = self.row_token.repeat(
            batch_size, num_rows, self.num_row_tokens, 1
        )  # (batch_size, num_rows, self.num_row_tokens, hidden_size)
        feature_embeddings = torch.cat(
            [feature_embeddings, current_row_tokens], dim=2
        )  # (batch_size, num_rows, num_columns + num_row_tokens, hidden_size)

        num_columns = feature_embeddings.shape[2]
        columns_to_attend = num_columns - self.num_row_tokens

        all_encoder_outputs = []
        for layer, compress_layer in zip(self.in_context_encoder, self.compression_layers):
            feature_embeddings = layer(feature_embeddings, num_context_rows, columns_to_attend)
            row_tokens_emb = feature_embeddings[:, :, -self.num_row_tokens :, :]

            # Each ROW token has its own downscaling linear projection
            per_row_token = torch.stack(
                [proj(row_tokens_emb[:, :, proj_idx, :]) for proj_idx, proj in enumerate(compress_layer)], dim=2
            )
            all_encoder_outputs.append(per_row_token)

        # sum all down-projected outputs from all encoder layers and up-project to the original hidden size
        encoder_outputs = torch.stack(all_encoder_outputs, dim=0).sum(dim=0)
        encoder_outputs = self.up_proj(encoder_outputs)
        return encoder_outputs

    def forward_decoder(
        self, target_embeddings: torch.Tensor, encoder_outputs: torch.Tensor, num_context_rows: int
    ) -> torch.Tensor:
        for layer in self.in_context_decoder:
            target_embeddings = layer(target_embeddings, num_context_rows, kv=encoder_outputs)
        return target_embeddings

    def apply_embeddings(self, data: dict[str, torch.Tensor], task_type: TaskType) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embeddings(data, task_type=task_type)
        return input_embeds[:, :, :-1, :], input_embeds[:, :, -1:, :]

    def forward(
        self,
        data: dict[str, torch.Tensor],
        task_type: TaskType,
        num_context_rows: int,
        return_encoder_output: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        feature_embeddings, target_embeddings = self.apply_embeddings(data, task_type)
        encoder_outputs = self.forward_encoder(feature_embeddings, num_context_rows)
        if return_encoder_output:
            return encoder_outputs[:, num_context_rows:]
        decoder_outputs = self.forward_decoder(target_embeddings, encoder_outputs, num_context_rows)
        return self.heads(decoder_outputs[:, num_context_rows:, -1], task_type == TaskType.REGRESSION)

    def forward_decoder_only(
        self, data: dict[str, torch.Tensor], task_type: TaskType, encoder_outputs: torch.Tensor, num_context_rows: int
    ) -> torch.Tensor:
        input_embeds = self.embeddings(data, task_type=task_type)  # (batch, max_rows, max_columns, hidden_size)
        target_embeddings = input_embeds[:, :, -1:, :]
        decoder_outputs = self.forward_decoder(target_embeddings, encoder_outputs, num_context_rows)
        return self.heads(decoder_outputs[:, num_context_rows:, -1], task_type == TaskType.REGRESSION)

    def forward_matching(
        self,
        data_left: dict[str, torch.Tensor],
        data_right: dict[str, torch.Tensor],
        num_context_rows: int,
        **kwargs,
    ) -> torch.Tensor:
        """Dual-table forward pass for entity matching.

        Both tables share the same target column (binary match label). The encoder
        runs over each side independently using shared weights; the per-side row
        tokens are then concatenated along the column axis so the decoder can
        cross-attend over both tables when producing the match prediction.

        When the two sides have the same number of feature columns, the encoder
        runs once over a (2, R, C, D) batch (fast path). When the schemas differ,
        each side is encoded separately and the row-token outputs are concatenated
        afterwards (fallback path) — same numerical result, no silent truncation.
        """
        feat_left, target_left = self.apply_embeddings(data_left, TaskType.CLASSIFICATION)
        feat_right, _target_right = self.apply_embeddings(data_right, TaskType.CLASSIFICATION)

        if feat_left.shape[2] == feat_right.shape[2]:
            # Fast path: equal column counts → single batched encoder pass over (2, R, C, D).
            feature_embeddings = torch.cat([feat_left, feat_right], dim=0)
            enc_left, enc_right = self.forward_encoder(feature_embeddings, num_context_rows)
            enc_left = enc_left.unsqueeze(0)
            enc_right = enc_right.unsqueeze(0)
        else:
            # Fallback: schemas differ → encode each side separately, concatenate row tokens on the batch axis.
            enc_left = self.forward_encoder(feat_left, num_context_rows)  # (1, R, num_row_tokens, D)
            enc_right = self.forward_encoder(feat_right, num_context_rows)  # (1, R, num_row_tokens, D)

        # Convert to correct shape
        # (2, R, num_row_tokens, D) -> (1, R, 2*num_row_tokens, D)
        encoder_outputs = torch.cat([enc_left, enc_right], dim=2)

        # Targets are by construction identical between the two sides; use one.
        decoder_outputs = self.forward_decoder(target_left, encoder_outputs, num_context_rows)
        return self.heads(decoder_outputs[:, num_context_rows:, -1], is_regression=False)


class OutlierHead(nn.Module):
    def __init__(self, config: FlexTabConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_head = RoundUpTo8LinearLayer(config.hidden_size, 1)

    def forward(self, encoder_outputs: torch.Tensor, is_regression: bool = False):
        out = self.dense(encoder_outputs)
        out = F.gelu(out)
        return self.output_head(out)


class FlexTabClustering(FlexTab):
    def __init__(self, clustering_out_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.clustering_out_dim = clustering_out_dim
        self.decoder_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.config.hidden_size * 4, clustering_out_dim),
        )
        del self.embeddings.is_missing_target_reg

    def construct_heads(self):
        # Do not construct any head here, it's called self.decoder_head...
        return None

    def forward(self, data: dict[str, torch.Tensor], num_context_rows: int, **kwargs) -> torch.Tensor:
        feature_embeddings, target_embeddings = self.apply_embeddings(data, TaskType.CLASSIFICATION)
        encoder_outputs = self.forward_encoder(feature_embeddings, num_context_rows)
        decoder_outputs = self.forward_decoder(target_embeddings, encoder_outputs, num_context_rows)
        h = decoder_outputs[:, :, -1, :]  # (batch, num_rows, hidden_size)
        out = self.decoder_head(h)
        return F.normalize(out, p=2, dim=-1)


class FlexTabAD(FlexTab):
    """Anomaly-detection variant of the FlexTab architecture.
    Uses label embeddings (0=inlier, 1=outlier, 2=unlabelled) as the decoder query,
    and a binary BCE head instead of the classification/regression head.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # label_embedding: 0=inlier, 1=outlier, 2=unlabelled
        self.label_embedding = nn.Embedding(3, self.config.hidden_size)
        nn.init.normal_(self.label_embedding.weight, std=1.0)
        nn.init.normal_(self.row_token, std=1)

    def construct_heads(self) -> nn.Module:
        return OutlierHead(self.config)

    def apply_embeddings(self, data: dict[str, torch.Tensor], task_type: TaskType) -> tuple[torch.Tensor, torch.Tensor]:
        feature_embeddings = self.embeddings(data, task_type=TaskType.ANOMALY_DETECTION)
        target_ids = data['target'].long()
        target_ids = torch.where(target_ids < 0, torch.full_like(target_ids, 2), target_ids).clamp(0, 2)
        target_embeddings = self.label_embedding(target_ids).unsqueeze(2)  # (batch_size, num_rows, 1, hidden_size)
        return feature_embeddings, target_embeddings
