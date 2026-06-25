import datetime
import os
from abc import ABC, abstractmethod
from typing import Collection, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow
import torch

from flextab.constants import (
    MAX_CLASSIFICATION_CLASSES,
    MAX_TEXT_LENGTH,
    NAN_BIN,
    NUM_REGRESSION_BINS_EXCLUDING_TAILS,
    TaskType,
    text_embedding_model_properties,
)
from flextab.data.sentence_embedder import SentenceEmbedder
from flextab.exceptions import FlexTabValueError
from flextab.utils.lru_cache import LRU_Cache


class Tokenizer(ABC):
    sentence_embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_dim = text_embedding_model_properties[sentence_embedding_model_name]['dimension']

    def __init__(self, random_seed=None, is_valid=False, is_drop_constant_columns=True):
        self.random_seed = random_seed
        self.is_valid = is_valid
        self.is_drop_constant_columns = is_drop_constant_columns

    @abstractmethod
    def texts_to_embeddings_dict(self, texts: Collection[str]) -> Tuple[dict[str, int], torch.Tensor]:
        pass

    def texts_to_tensor(
        self, texts: Collection[str], text_to_idx: Dict[str, int], text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if len(texts) == 0:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float16)

        indices = torch.tensor([text_to_idx[text[:MAX_TEXT_LENGTH]] for text in texts], dtype=torch.int)
        return text_embeddings[indices]

    @staticmethod
    def value_or_nan(values: Union[pd.Series, np.ndarray]):
        if isinstance(values, pd.Series):
            values = values.values
        return torch.tensor(np.where(np.isnan(values), np.array(0), values + 1))

    def quantize_column(self, y_context: pd.Series, y_query: Optional[pd.Series] = None):
        """Creates NUM_REGRESSION_BINS_EXCLUDING_TAILS + 2 bins based on the following quantiles:
        q_0, q_{1/2n}, q_{3/2n}, ..., q_{(2n-1)/2n}, q_1
        where n = NUM_REGRESSION_BINS_EXCLUDING_TAILS. Quantiles are computed on y_context only.
        The result is then applied to y_context (and y_query if provided).
        It returns:
        - lower_bound_index: in [0, n]. Numbers below q_{1/2n} will have index 0, numbers between q_{1/2n} and q_{3/2n}
            will have index 1, etc., up to index n for numbers above q_{(2n-1)/2n}.
        - delta: in [0, 1] that gives the position in the interval; so the "real" bin is lower_bound_index + delta in [0, n + 1]
        - quantiles: the reference values above, for reconstructing the original value and for inference.
        """
        a = np.asarray(y_context.values)
        b = a if y_query is None else np.concatenate([a, np.asarray(y_query.values)])
        # Drop NaN otherwise all results are NaN
        a = a[np.isfinite(a)]
        if len(a) == 0:
            raise FlexTabValueError('No finite regression target, cannot proceed.')
        n = NUM_REGRESSION_BINS_EXCLUDING_TAILS

        q = np.linspace(1 / (2 * n), (2 * n - 1) / (2 * n), n)
        quantiles = np.quantile(a, q)

        # Clip: even though we try to not clip anything at all, in practice that's a bad idea,
        # because a single _extreme_ outlier can distort all predictions.
        # E.g. in task `os__texttab__it_salary__default__yearly brutto salary (without bonus and stocks) in eur`
        # normal values go up to 200k, but there is a single outlier having value 9_999_999_999.
        # Even if the tail bin gets a tiny probability, when taking the scalar product
        # np.sum(probabilities * bin_centers)
        # that's more than enough to skew all values towards that extreme outlier.
        # So here we clip values forcing max(distribution) - 99% quantile not to be larger
        # than 99% quantile - median (and similarly on the left)
        # That's quite a permissive clipping anyway...
        median = np.median(a)
        right_margin = quantiles[-1] - median
        largest_value = min(np.max(a), quantiles[-1] + right_margin)
        left_margin = median - quantiles[0]
        smallest_value = max(np.min(a), quantiles[0] - left_margin)
        quantiles = np.concatenate([[smallest_value], quantiles, [largest_value]])  # Length: n+2

        # Digitize b to find which interval each value belongs to
        indices = np.digitize(b, quantiles) - 1
        # The above values are in [-1, n + 1], because digitize returns 0 or len(quantiles) = n+2
        # for values outside the range (i.e. extreme outliers)

        indices = np.clip(indices, 0, n)

        # Compute delta
        lower_bounds = quantiles[indices]
        upper_bounds = quantiles[indices + 1]
        delta = (b - lower_bounds) / np.maximum(upper_bounds - lower_bounds, 1e-10)
        delta = np.clip(delta, 0, 1)

        return indices, delta, quantiles

    def build_classification_labels_one_column(self, y_context: pd.Series):
        """A bit like OrdinalEncoder fit on train, but for test, if we find an unseen label,
        instead of raising a ValueError, we assign an integer larger than the largest seen label.
        This avoids the (tiny) information leakage that if e.g. we see only labels 0 and 2 in train,
        then we know there is at least one row in test with label 1.
        """
        sorted_value_to_count = y_context.value_counts(sort=False)
        # First sort alphabetically by the labels, and then sort by count (descending),
        # using stable sort to preserve alphabetical order for equal counts
        sorted_value_to_count = sorted_value_to_count.sort_index().sort_values(ascending=False, kind='stable')

        # get most frequent `MAX_CLASSIFICATION_CLASSES` labels and their counts
        sorted_value_to_count = sorted_value_to_count.iloc[:MAX_CLASSIFICATION_CLASSES]

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        shuffled_labels = np.random.permutation(sorted_value_to_count.index)

        label_classes = list(shuffled_labels)

        # labels get indices from 0 to MAX_CLASSIFICATION_CLASSES - 1 (inclusive)
        # index MAX_CLASSIFICATION_CLASSES is reserved for everything else
        # in masking, those indices are increased by 1 to indicate that 0 is a mask
        # and then MAX_CLASSIFICATION_CLASSES + 1 is the max index
        labels_idx = np.arange(0, len(label_classes))
        label_to_index = {l: idx for l, idx in zip(label_classes, labels_idx)}
        result = np.asarray([label_to_index.get(y, MAX_CLASSIFICATION_CLASSES) for y in y_context.values])
        return result, np.asarray(label_classes)

    @staticmethod
    def time_to_seconds(t: Union[datetime.time, pyarrow.time64]):
        try:
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6
        except Exception:
            if t is not None:
                print('Expected time found', type(t), t)
            return np.nan

    def convert_time_to_seconds(self, context_df, query_df, column_name):
        context_df[column_name] = context_df[column_name].apply(self.time_to_seconds)
        query_df[column_name] = query_df[column_name].apply(self.time_to_seconds)

    def convert_to_datetime(self, context_df, query_df, column_name):
        context_df[column_name] = pd.to_datetime(context_df[column_name], errors='coerce')
        query_df[column_name] = pd.to_datetime(query_df[column_name], errors='coerce')

    def convert_type_(self, context_df: pd.DataFrame, query_df: pd.DataFrame, column_name: str):
        dt = str(context_df[column_name].dtype)
        if dt.startswith('time64') or (dt == 'object' and isinstance(context_df[column_name].iloc[0], datetime.time)):
            # time type; either pyarrow.time64[us] (whole column dtype) or datetime.time (column is object)
            self.convert_time_to_seconds(context_df, query_df, column_name)
            dt = 'float64'
        elif (
            dt.startswith('date')
            or dt.startswith('timestamp')
            or (dt == 'object' and isinstance(context_df[column_name].iloc[0], datetime.date))
        ):
            # date type; either pyarrow.date32[day] (whole column dtype) or datetime.date (column is object)
            self.convert_to_datetime(context_df, query_df, column_name)
            dt = 'datetime64[ns]'
        elif dt.split('[')[0].lower() not in {
            'int64',
            'int32',
            'int16',
            'int8',
            'uint64',
            'uint32',
            'uint16',
            'uint8',
            'float64',
            'float32',
            'float16',
            'datetime64',
            'double',
            'date32',
            'timestamp',
        }:
            if dt not in ['bool', 'string[pyarrow]', 'bool[pyarrow]', 'category', 'string', 'object']:
                print(f'Data type {dt} not recognized! Defaulting to string')
            elif dt == 'object' and not isinstance(context_df[column_name].iloc[0], str):
                is_null = context_df[column_name].isnull()
                if not is_null.iloc[0]:
                    value = context_df[column_name].iloc[0]
                elif is_null.all():
                    value = None
                else:
                    value = context_df[column_name][~is_null].iloc[0]
                if isinstance(value, datetime.date):
                    self.convert_to_datetime(context_df, query_df, column_name)
                elif isinstance(value, datetime.time):
                    self.convert_time_to_seconds(context_df, query_df, column_name)
                elif not isinstance(value, (str, bool)):
                    print(f'Warning, dtype is object, but first non-null value is {type(value)}. Converting to str.')
                    print(context_df[column_name].head())
            dt = 'object'
        return dt

    def process_target(self, data, y_context: pd.Series, task_type: TaskType):
        total_length = data['number_percentile_floor'].shape[0]
        # Initialize all target rows to -100 (masked); only context rows get real values below
        data['target'] = torch.full((total_length,), -100, dtype=torch.float32)
        data['target_delta'] = torch.zeros((total_length,), dtype=torch.float32)
        if task_type == TaskType.REGRESSION:
            y_context = y_context.astype(float)
            labels_lower_bin, delta_labels, label_classes = self.quantize_column(y_context)
            assert np.max(labels_lower_bin) <= NUM_REGRESSION_BINS_EXCLUDING_TAILS
            data['target'][: len(y_context)] = torch.tensor(labels_lower_bin, dtype=torch.float32)
            data['target_delta'][: len(y_context)] = torch.tensor(delta_labels, dtype=torch.float32)
        elif task_type == TaskType.CLASSIFICATION:
            labels_lower_bin, label_classes = self.build_classification_labels_one_column(y_context)
            data['target'][: len(y_context)] = torch.tensor(labels_lower_bin, dtype=torch.float32)
        else:
            raise FlexTabValueError(f"Unsupported task type '{task_type}'")

        context_size = len(y_context)
        data['text_embeddings'][context_size:, -1:] = 0
        return data, {y_context.name: label_classes}

    def replace_inf_values(self, column_values: pd.Series):
        array_values = column_values.values
        if not np.isfinite(array_values).any():
            clipped_values = np.full(array_values.shape, np.nan)
        else:
            max_value = array_values[np.isfinite(array_values)].max()
            min_value = array_values[np.isfinite(array_values)].min()
            clipped_values = np.clip(array_values, min_value - 1, max_value + 1)
        return pd.Series(clipped_values, index=column_values.index)

    def process_features(self, X_context, X_query, data, text_to_idx, text_embeddings):
        total_length = len(X_context) + len(X_query)
        for column_index, c in enumerate(X_context.columns):
            str_dtype = self.convert_type_(X_context, X_query, c)
            column_values = pd.concat([X_context[c], X_query[c]])

            if str_dtype == 'object':
                data['text_embeddings'][:, column_index] = self.texts_to_tensor(
                    column_values.astype(str), text_to_idx, text_embeddings
                )
            elif str_dtype.split('[')[0] in ['datetime64', 'date32', 'timestamp']:
                data['date_year_month_day_weekday'][:, column_index, 0] = self.value_or_nan(
                    column_values.dt.year.clip(2000, 2050) - 2000
                ).int()
                data['date_year_month_day_weekday'][:, column_index, 1] = self.value_or_nan(
                    column_values.dt.month - 1
                ).int()
                data['date_year_month_day_weekday'][:, column_index, 2] = self.value_or_nan(
                    column_values.dt.day - 1
                ).int()
                data['date_year_month_day_weekday'][:, column_index, 3] = self.value_or_nan(
                    column_values.dt.weekday
                ).int()
            else:
                del column_values
                context_values = X_context[c].astype(float)
                query_values = X_query[c].astype(float)
                data = self._process_numeric_feature(
                    data=data,
                    context_values=context_values,
                    query_values=query_values,
                    column_index=column_index,
                    total_length=total_length,
                )

        return data

    def _process_numeric_feature(
        self, data: dict, context_values: pd.Series, query_values: pd.Series, column_index: int, total_length: int
    ) -> dict:
        column_labels_lower_bin = np.zeros(total_length, dtype=int)
        column_delta_labels = np.zeros(total_length, dtype=float)
        # np.quantile doesn't like infinity for some reason.
        # Since exact values don't matter, we just replace inf with max+1 and -inf with min-1.
        context_values = self.replace_inf_values(context_values)
        query_values = self.replace_inf_values(query_values)
        nan_mask = pd.concat([context_values.isna(), query_values.isna()]).values

        if (~nan_mask).any():
            column_labels_lower_bin[~nan_mask], column_delta_labels[~nan_mask], _ = self.quantize_column(
                context_values.dropna(), query_values.dropna()
            )
        # assign nan value to the last bin with index `NUM_REGRESSION_BINS`
        column_labels_lower_bin[nan_mask] = NAN_BIN

        data['number_percentile_floor'][:, column_index] = torch.tensor(column_labels_lower_bin)
        data['number_percentile_delta'][:, column_index] = torch.tensor(column_delta_labels)
        return data

    def __call__(
        self,
        X_context: pd.DataFrame,
        y_context: Optional[pd.Series],
        X_query: pd.DataFrame,
        task_type: Optional[TaskType],
    ) -> Tuple[dict, dict]:
        """Tokenizes features and (if y_context is provided) the target column.

        When y_context is None, only feature tensors are returned and
        the caller is responsible for setting data['target'] to label indices afterwards.

        Returns:
        - data: dict of torch tensors
        - label_classes: dict mapping target column name → classes array (empty when y_context is None)
        """
        if self.is_drop_constant_columns:
            X_context = X_context.dropna(axis=1, how='all').copy()
        X_query = X_query[X_context.columns].copy()

        total_length = len(X_context) + len(X_query)

        if y_context is not None:
            num_columns = len(X_context.columns) + 1
            col_names = [str(x) for x in X_context.columns] + [str(y_context.name)]
        else:
            num_columns = len(X_context.columns)
            col_names = [str(x) for x in X_context.columns]

        # collect all texts to embed (column names + string cell values)
        all_texts = col_names[:]
        for c in X_context.columns:
            str_dtype = self.convert_type_(X_context, X_query, c)
            if str_dtype == 'object':
                all_texts.extend(X_context[c].astype(str).tolist())
                all_texts.extend(X_query[c].astype(str).tolist())
        text_to_idx, text_embeddings = self.texts_to_embeddings_dict(all_texts)

        data = {
            'column_embeddings': self.texts_to_tensor(col_names, text_to_idx, text_embeddings),
            'text_embeddings': torch.zeros((total_length, num_columns, self.embedding_dim), dtype=torch.float16),
            'date_year_month_day_weekday': torch.zeros((total_length, num_columns, 4), dtype=torch.int64),
            'number_percentile_floor': torch.full((total_length, num_columns), dtype=torch.int64, fill_value=-100),
            'number_percentile_delta': torch.zeros((total_length, num_columns), dtype=torch.float32),
        }

        all_label_classes = {}
        if y_context is not None:
            data, all_label_classes = self.process_target(data, y_context, task_type)

        data = self.process_features(X_context, X_query, data, text_to_idx, text_embeddings)
        return data, all_label_classes


class InferenceTokenizer(Tokenizer):
    def __init__(
        self, random_seed=None, is_valid=False, is_drop_constant_columns=True, sentence_embedder_batch_size: int = 512
    ):
        super().__init__(random_seed, is_valid, is_drop_constant_columns)

        self.sentence_embedder = SentenceEmbedder(
            self.sentence_embedding_model_name,
            batch_size=sentence_embedder_batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.cache = LRU_Cache(max_size=int(os.getenv('LRU_CACHE_SIZE', 100_000)))

    def texts_to_embeddings_dict(self, texts: Collection[str]) -> Tuple[dict[str, int], torch.Tensor]:
        # We sort them by length, so that the tokenizer can batch them efficiently
        # with minimal pad
        texts = [text[:MAX_TEXT_LENGTH] for text in set(texts)]
        sorted_unique_texts = sorted(set(texts), key=len, reverse=True)
        text_to_idx = {text: i for i, text in enumerate(sorted_unique_texts)}

        # Check if we have the embeddings in the cache
        result_dict = {text: self.cache[text] for text in sorted_unique_texts}
        missing_texts = [text for text, result in result_dict.items() if result is None]
        # Embed the missing texts and store in cache
        if missing_texts:
            missing_embeddings = self.sentence_embedder.embed(missing_texts)
            for i, text in enumerate(missing_texts):
                emb = missing_embeddings[i]
                self.cache[text] = emb
                result_dict[text] = emb

        return text_to_idx, torch.stack(list(result_dict.values()))
