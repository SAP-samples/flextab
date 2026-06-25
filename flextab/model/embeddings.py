from typing import Dict

import torch
from torch import nn

from flextab.constants import (
    MAX_CLASSIFICATION_CLASSES,
    NAN_BIN,
    TOTAL_NUM_REGRESSION_BINS,
    TaskType,
    text_embedding_model_properties,
)
from flextab.data.tokenizer import Tokenizer


class DateEmbeddings(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.year_embeddings = nn.Embedding(52, hidden_size)
        self.month_embeddings = nn.Embedding(13, hidden_size)
        self.day_embeddings = nn.Embedding(32, hidden_size)
        self.weekday_embeddings = nn.Embedding(8, hidden_size)

        # Custom weight initialization: instead of the usual std=1.0, we use std=0.5,
        # because we then sum the 4 objects together and want the result to have std=1.0
        nn.init.normal_(self.year_embeddings.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.month_embeddings.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.day_embeddings.weight, mean=0.0, std=0.5)
        nn.init.normal_(self.weekday_embeddings.weight, mean=0.0, std=0.5)

        # Also, we initialize the embedding for "unknown" (index 0) to zero,
        # so that non-dates are centered around 0.
        with torch.no_grad():
            self.year_embeddings.weight[0].fill_(0)
            self.month_embeddings.weight[0].fill_(0)
            self.day_embeddings.weight[0].fill_(0)
            self.weekday_embeddings.weight[0].fill_(0)

    def forward(self, date_year_month_day_weekday):
        # date_year_month_day_weekday has shape (num_rows, num_cols, 4)
        embeds = self.year_embeddings(date_year_month_day_weekday[..., 0])
        embeds += self.month_embeddings(date_year_month_day_weekday[..., 1])
        embeds += self.day_embeddings(date_year_month_day_weekday[..., 2])
        embeds += self.weekday_embeddings(date_year_month_day_weekday[..., 3])

        return embeds


class SafeLayerNorm(nn.LayerNorm):
    """LayerNorm implementation has a bug on CUDA (likely due to 32-bit indexing) when the number of elements is >= 2**32.
    See https://github.com/pytorch/pytorch/issues/149301
    This workaround splits the input into smaller batches in that case.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.numel() >= 2**32:
            input_shape = input.shape
            input = input.view(-1, input_shape[-1])
            output = torch.zeros_like(input)
            batch_len = int((2**32 - 1) / input.shape[-1])
            for i in range(0, input.shape[0], batch_len):
                output[i : i + batch_len] = super().forward(input[i : i + batch_len])
            return output.view(input_shape)
        return super().forward(input)


class CellEmbeddings(nn.Module):
    """Embedding module for self supervised learning.
    On the input side, it sums four contributions:
    - Numbers (itself coming from embedding three one-hot encoded values: sign, exponent, fraction)
    - Dates (itself coming from embedding four one-hot encoded values: year, month, day, weekday, plus one multi-hot encoded: holidays)
    - Column names (sentence embedding of the column name, adjusted to the hidden size)
    - (String) contents (sentence embedding of the column name, adjusted to the hidden size)
    For labels, it also computes and returns the sentence embedding of the string contents (not adjusted to the size).
    All string embeddings (column names, contents of both input and labels)
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        # TOTAL_NUM_REGRESSION_BINS = NUM_REGRESSION_BINS_EXCLUDING_TAILS + 3, two to store the "tail" bins plus one extra bin for nan
        self.number_embeddings = nn.Embedding(TOTAL_NUM_REGRESSION_BINS, config.hidden_size)

        # TOTAL_NUM_REGRESSION_BINS = NUM_REGRESSION_BINS_EXCLUDING_TAILS + 3, two to store the "tail" bins plus one
        # extra bin for mask (we do not have nans in target column)
        self.target_embedding_layer_reg = nn.Embedding(TOTAL_NUM_REGRESSION_BINS, config.hidden_size)
        self.is_missing_target_reg = nn.Parameter(torch.zeros(1, 1, config.hidden_size))  # for missing target flag
        # MAX_CLASSIFICATION_CLASSES plus one extra bin for everything else and one extra for mask
        self.target_embedding_layer_classif = nn.Embedding(MAX_CLASSIFICATION_CLASSES + 2, config.hidden_size)

        self.date_embeddings = DateEmbeddings(config.hidden_size)

        self.column_remapping = nn.Linear(Tokenizer.embedding_dim, config.hidden_size)
        self.content_remapping = nn.Linear(Tokenizer.embedding_dim, config.hidden_size)

        # Modify the initialization of the linear layers: they are created supposing the input has
        # std=1.0, but instead we feed embeddings in, and they have _norm_ 1.0, i.e. std=sqrt(1/emb_size).
        # So we need to increase the weights by sqrt(emb_size).
        # By default, it uses Kaiming-He, which is Uniform[-1/sqrt(fan_in), 1/sqrt(fan_in)],
        # and here fan_in = emb_size, so in the end all we need is uniform[-1, 1]!
        inverse_average_norm = (
            1 / text_embedding_model_properties[Tokenizer.sentence_embedding_model_name]['average_norm']
        )
        nn.init.uniform_(self.column_remapping.weight, a=-inverse_average_norm, b=inverse_average_norm)
        nn.init.uniform_(self.content_remapping.weight, a=-inverse_average_norm, b=inverse_average_norm)
        # Also, initialize the bias to zero, so that non-text cells are centered around zero
        nn.init.zeros_(self.column_remapping.bias)
        nn.init.zeros_(self.content_remapping.bias)

        self.layer_norm = SafeLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def increase_by_one_and_map_negative_to_zero(self, tensor):
        """In dataset, "valid" labels are 0, 1, 2, ... and masked values are -100.
        We want to map them to 1, 2, 3, ... and 0.
        (Also, they might be float values, but we want to keep them as integers.)
        """
        tensor = tensor.int()
        return torch.where(tensor < 0, torch.zeros_like(tensor), tensor + 1)

    def get_target_embeddings(self, target, target_delta, is_classification, dtype):
        if is_classification:
            target_values_classif = self.increase_by_one_and_map_negative_to_zero(target)
            target_embeds_classif = self.target_embedding_layer_classif(
                target_values_classif
            )  # (batch_size, rows, embed_dim)
            target_embeds = target_embeds_classif.type(dtype)
        else:
            # regression
            missing_mask = torch.isnan(target)  # (batch_size, num_rows)
            target = torch.where(missing_mask, torch.zeros_like(target), target)
            target_delta = torch.where(missing_mask, torch.zeros_like(target_delta), target_delta)

            target_values_reg = self.increase_by_one_and_map_negative_to_zero(target)
            target_embeds_reg = self.target_embedding_layer_reg(target_values_reg)
            target_plus_one_embeds_reg = self.target_embedding_layer_reg(target_values_reg + 1)
            delta = target_delta.type(target_embeds_reg.dtype).unsqueeze(-1)
            target_embeds_reg = target_embeds_reg * (1 - delta) + target_plus_one_embeds_reg * delta
            target_embeds = torch.where(
                missing_mask.unsqueeze(-1), self.is_missing_target_reg.type(dtype), target_embeds_reg
            )
        return target_embeds

    def forward(self, input_dict: Dict, task_type: TaskType):
        batch_size, num_rows, num_cols, _ = input_dict['text_embeddings'].shape

        number_perc_floor = input_dict['number_percentile_floor']
        input_embeds = torch.zeros(
            (batch_size, num_rows, num_cols, self.hidden_size),
            dtype=self.number_embeddings.weight.dtype,
            device=self.number_embeddings.weight.device,
        )
        mask = number_perc_floor > -99
        delta = input_dict['number_percentile_delta'][mask].type(input_embeds.dtype).unsqueeze(-1)
        input_embeds[mask] = self.number_embeddings(number_perc_floor[mask]) * (1 - delta)

        # last bin is NAN_BIN, we clamp at NAN_BIN - 1 to avoid spillovers interpolating between "max" and "nan"...
        next_perc = torch.minimum(number_perc_floor[mask] + 1, torch.tensor(NAN_BIN - 1))
        input_embeds_plus_one = self.number_embeddings(next_perc) * delta
        input_embeds[mask] += input_embeds_plus_one

        input_embeds += self.date_embeddings(
            input_dict['date_year_month_day_weekday']
        )  # (batch_size, rows, cols, embed_dim)

        unsqueezed = input_dict['column_embeddings'].unsqueeze(1)  # (batch_size, 1, cols, embed_dim)
        input_embeds += self.column_remapping(unsqueezed.type(self.column_remapping.weight.dtype))

        # FlexTab path: zero out text embeddings for target (last column)
        text_embeddings = torch.cat(
            [input_dict['text_embeddings'][:, :, :-1], torch.zeros_like(input_dict['text_embeddings'][:, :, -1:])],
            dim=2,
        ).type(self.content_remapping.weight.dtype)

        content_embeds = self.content_remapping(text_embeddings)
        input_embeds += content_embeds
        del content_embeds

        is_classification = task_type == TaskType.CLASSIFICATION
        input_embeds[:, :, -1] += self.get_target_embeddings(
            input_dict['target'], input_dict['target_delta'], is_classification, input_embeds.dtype
        )

        input_embeds = self.layer_norm(input_embeds)
        input_embeds = self.dropout(input_embeds)
        return input_embeds
