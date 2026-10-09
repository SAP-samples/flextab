from __future__ import annotations

import logging
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from tqdm import trange

from flextab.constants import TaskType
from flextab.data.collate import collate_stack
from flextab.data.tokenizer import Tokenizer
from flextab.exceptions import FlexTabInputValidationError
from flextab.model.gpu_utils import retry_on_cuda_oom
from flextab.model.torch_model import FlexTab, FlexTabAD, FlexTabClustering
from flextab.utils.to_device import to_device

if TYPE_CHECKING:
    from flextab.flextab_multi.relbench_data import RelbenchData

HERE = pathlib.Path(__file__).parent
HF_REPO_ID = 'SAP/FlexTab'
logger = logging.getLogger(__name__)


class FlexTabModel(BaseEstimator):
    """Class to perform both classification and regression with FlexTab.
    The subclasses FlexTabClassifier and FlexTabRegressor are just thin wrapper over this.
    checkpoint: path to the checkpoint file; must be of size ModelSize (defaults to flextab/model/ckpt.pt)
    bagging: number of bagging iterations; if 1 there is no bagging.
    max_context_size: maximum number of samples to use for training
    """

    MAX_NUM_COLUMNS = 500

    def __init__(
        self,
        checkpoint: str = 'flextab_v1_0_0.pt',
        bagging: int = 8,
        max_context_size: int = 8192,
        is_drop_constant_columns: bool = True,
        sentence_embedder_batch_size: int = 512,
        device: str | None = None,
    ):
        self.checkpoint = hf_hub_download(repo_id=HF_REPO_ID, filename=checkpoint)
        self.bagging = bagging
        self.max_context_size = max_context_size
        logger.info(f'Using max_context_size={max_context_size}')
        # We're using a single GPU here, even if more are available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        # Load weights first to know how to embed the numbers
        state_dict = torch.load(self.checkpoint, map_location=self.device, weights_only=True)

        self.model = FlexTab()
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability(0)[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
            self.model = self.model.to(dtype=self.dtype)
        else:
            self.dtype = torch.float32

        self.model.load_weights(state_dict, self.device)
        self.seed = 42
        self.is_drop_constant_columns = is_drop_constant_columns
        self.tokenizer = Tokenizer(
            random_seed=self.seed, sentence_embedder_batch_size=sentence_embedder_batch_size, is_valid=True
        )
        self.model.to(self.device).eval()
        self.sentence_embedder_batch_size = sentence_embedder_batch_size
        self.X_: None | pd.DataFrame = None
        self.y_: None | pd.Series = None
        self.task_type: None | TaskType = None
        self.classes_: None | np.ndarray = None
        self.target_name: None | str = None

    def __sklearn_is_fitted__(self):
        return self.y_ is not None

    @staticmethod
    def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.columns.duplicated().any():
            df.columns = [c + f'_{i}' for i, c in enumerate(df.columns)]
        return df

    def fit(self, X: pd.DataFrame, y: pd.Series, task_type: TaskType):
        """Fit (i.e. memorize) one context dataframe X and a single target column y.
        task_type: TaskType.CLASSIFICATION or TaskType.REGRESSION
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index, name='TARGET')

        self.bagging_config = self.bagging
        if X.shape[0] < self.max_context_size:
            self.bagging_config = 1

        if (X.index != y.index).any():
            raise FlexTabInputValidationError('X and y must have the same index')

        missing_mask = y.isnull()
        if missing_mask.any():
            X = X[~missing_mask]
            y = y[~missing_mask]

        if X.index.duplicated().any():
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
        X = self.deduplicate_columns(X)

        self.X_ = X
        self.y_ = y
        self.task_type = task_type
        self.target_name = y.name if y.name is not None else 'TARGET'

        if task_type == TaskType.CLASSIFICATION:
            self.classes_ = unique_labels(y)
        else:
            self.y_ = y.astype(float)
            self.classes_ = None

        return self

    def get_one_tokenized_data(self, X_test, bagging_index):
        assert isinstance(self.X_, pd.DataFrame)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=self.X_.columns)

        X_test = self.deduplicate_columns(X_test)
        X_train = self.X_
        if not (X_train.columns == X_test.columns).all():
            raise FlexTabInputValidationError('Columns of X_test do not match columns of X_train')

        if self.bagging_config > 1:
            X_train = X_train.sample(self.max_context_size, replace=False, random_state=self.seed + bagging_index)
        elif len(X_train) > self.max_context_size:
            X_train = X_train.sample(self.max_context_size, replace=False, random_state=self.seed + bagging_index)

        if self.is_drop_constant_columns:
            constant_cols = list(X_train.columns[X_train.nunique() == 1])
            if constant_cols:
                X_train = X_train.drop(columns=constant_cols)

        if X_train.shape[1] + 1 > self.MAX_NUM_COLUMNS:
            X_train = X_train[sorted(X_train.columns)]
            X_train = X_train.sample(
                n=self.MAX_NUM_COLUMNS - 1, axis=1, random_state=self.seed + bagging_index, replace=False
            )

        X_test = X_test[X_train.columns]
        self.tokenizer.random_seed = self.seed + bagging_index
        data, label_classes = self.tokenizer(X_train, self.y_.loc[X_train.index], X_test, self.task_type)

        result = {
            'data': data,
            'num_query_rows': X_test.shape[0],
            'num_context_rows': X_train.shape[0],
            'num_features': X_train.shape[1],
            'task_type': self.task_type,
            'label_classes': {k: np.asarray(v) for k, v in label_classes.items()},
        }
        return collate_stack([result])  # to have batch of size 1

    def _extract_subset_tokenized_data_and_send_to_device(
        self, tokenized_data, start: int, query_chunk_size: int
    ) -> dict:
        """Given tokenized data for all queries and context from a single bagging fold,
        extract a subset containing all context and only a chunk of the queries.
        Also sends the data to the correct device.
        """
        # Duplicate all metadata: num_query_rows, num_cols, task_type, label_classes, num_context_rows
        context_size = tokenized_data['num_context_rows']
        subset = {k: v if k != 'data' else {} for k, v in tokenized_data.items()}
        # For data, we need to include context and only part of the queries
        for key, value in tokenized_data['data'].items():
            # For column embeddings, there are no rows, so we copy
            if key == 'column_embeddings':
                subset['data'][key] = value
            else:
                context_rows = tokenized_data['data'][key][:, :context_size]
                these_query_rows = tokenized_data['data'][key][
                    :, context_size + start : context_size + start + query_chunk_size
                ]
                subset['data'][key] = torch.cat([context_rows, these_query_rows], dim=1)

        return to_device(subset, self.device, raise_on_unexpected=False)

    def ensure_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)
        return X

    def predict(self, X: pd.DataFrame, query_chunk_size: int = 1024) -> np.ndarray:
        """For regression, returns scalar predictions.
        For classification, returns the most likely class labels.
        """
        result = self.predict_proba(X, query_chunk_size=query_chunk_size)
        if self.task_type == TaskType.CLASSIFICATION:
            predicted_indices = result.argmax(-1)
            return self.classes_[predicted_indices]
        return result

    def _predict_tokenized(self, tokenized_data) -> torch.Tensor:
        """Run model on tokenized chunk, return query-only predictions."""
        output = self.model(**tokenized_data).squeeze(0)
        label_classes = tokenized_data['label_classes'][self.target_name]

        if self.task_type == TaskType.REGRESSION:
            preds, _ = self.model.extract_prediction_regression(output, label_classes)
        else:
            _, unordered_preds = self.model.extract_prediction_classification(output, label_classes)
            preds = self.reorder_logits(unordered_preds, label_classes, self.classes_)
        return preds

    def _predict_on_chunks(
        self, X: pd.DataFrame, bagging_index: int, query_chunk_size: int, predict_fn
    ) -> torch.Tensor:
        """Tokenize, chunk queries, run predict_fn on each chunk, and concatenate."""
        tokenized_data_one_bag = self.get_one_tokenized_data(X, bagging_index)
        results = []
        for i in range(0, tokenized_data_one_bag['num_query_rows'], query_chunk_size):
            chunk = self._extract_subset_tokenized_data_and_send_to_device(tokenized_data_one_bag, i, query_chunk_size)
            results.append(predict_fn(chunk))
        return torch.cat(results, dim=0)

    @retry_on_cuda_oom(default_size=1024, param_name='query_chunk_size')
    @torch.inference_mode()
    def predict_proba(
        self,
        X: pd.DataFrame,
        query_chunk_size: int = 1024,
    ) -> np.ndarray:
        """For classification, returns probability distributions over classes.
        For regression, returns scalar predictions.
        """
        check_is_fitted(self)
        X = self.ensure_dataframe(X)

        predictions_for_all_bags = []
        for bagging_index in range(self.bagging_config):
            predictions_for_all_bags.append(
                self._predict_on_chunks(X, bagging_index, query_chunk_size, self._predict_tokenized)
            )

        return self._aggregate_from_bags(predictions_for_all_bags)

    def _aggregate_from_bags(self, predictions_for_all_bags: list[torch.Tensor]) -> np.ndarray:
        """predictions_for_all_bags: list of tensors, one per bagging fold.
        Each tensor has shape (num_query_rows,) for regression or (num_query_rows, num_classes) for classification.
        Returns aggregated predictions as numpy array.
        """
        stacked = torch.stack(predictions_for_all_bags)
        if self.task_type == TaskType.CLASSIFICATION:
            all_probs = torch.nn.functional.softmax(stacked, dim=-1)
            preds = all_probs.mean(dim=0)
        else:
            preds = torch.mean(stacked, dim=0)
        return preds.cpu().numpy()

    @staticmethod
    def reorder_logits(logits, tokenized_classes, classes) -> torch.Tensor:
        class_mapping = {cls: idx for idx, cls in enumerate(classes)}
        new_logits = torch.full(
            (logits.shape[0], len(classes)), float('-inf'), dtype=logits.dtype, device=logits.device
        )

        if len(tokenized_classes) == 1:
            tokenized_classes = tokenized_classes[0]

        if tokenized_classes.ndim > 1:
            for batch_idx in range(logits.shape[0]):
                batch_classes = tokenized_classes[batch_idx]
                indices = np.array([class_mapping[cls] for cls in batch_classes if cls in class_mapping])
                new_logits[batch_idx, indices] = logits[batch_idx, : len(indices)]
        else:
            indices = np.array([class_mapping[cls] for cls in tokenized_classes])
            new_logits[:, indices] = logits[:, : len(tokenized_classes)]
        return new_logits


class FlexTabClassifier(ClassifierMixin, FlexTabModel):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        return super().fit(X, y, task_type=TaskType.CLASSIFICATION)


class FlexTabRegressor(RegressorMixin, FlexTabModel):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        return super().fit(X, y, task_type=TaskType.REGRESSION)


class FlexTabClusterer(TransformerMixin, BaseEstimator):
    """Transductive clusterer using FlexTab.

    fit(X, y=None): no-op (returns self). All of X serves as both context and query at transform time.
    transform(X): returns L2-normalized embeddings for all rows in X, shape (n_rows, clustering_out_dim).
    predict(X, n_clusters): transform(X) followed by K-means.
    """

    def __init__(
        self,
        checkpoint: str = 'flextab_clustering_v1_0_0.pt',
        sentence_embedder_batch_size: int = 512,
        max_context_size: int = 8192,
        max_columns: int = 250,
        seed: int = 42,
    ):
        self.checkpoint = hf_hub_download(repo_id=HF_REPO_ID, filename=checkpoint)
        self.sentence_embedder_batch_size = sentence_embedder_batch_size
        self.max_context_size = max_context_size
        self.max_columns = max_columns
        self.seed = seed

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_dict = torch.load(self.checkpoint, map_location=self.device, weights_only=True)

        decoder_head_key = 'decoder_head.2.weight'
        if decoder_head_key not in state_dict and 'module.' + decoder_head_key in state_dict:
            decoder_head_key = 'module.' + decoder_head_key
        clustering_out_dim = state_dict[decoder_head_key].shape[0]

        self.model = FlexTabClustering(clustering_out_dim=clustering_out_dim)
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
            self.model = self.model.to(dtype=self.dtype)
        else:
            self.dtype = torch.float32

        self.model.load_weights(state_dict, self.device)
        self.model.to(self.device).eval()

        self.tokenizer = Tokenizer(sentence_embedder_batch_size=sentence_embedder_batch_size, is_valid=True)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    @torch.inference_mode()
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if X.shape[1] > self.max_columns:
            X = X.sample(n=self.max_columns, axis=1, random_state=self.seed)

        if len(X) > self.max_context_size:
            X = X.sample(n=self.max_context_size, axis=0, random_state=self.seed)

        X = X.reset_index(drop=True)
        X_query_dummy = X.iloc[:1]

        batch, num_context_rows = self._prepare_batch(X, X_query_dummy)
        out = self.model(data=batch['data'], num_context_rows=batch['num_context_rows'], task_type=TaskType.CLUSTERING)
        return out[0, :num_context_rows, :].cpu().float().numpy()

    def _prepare_batch(self, X_context: pd.DataFrame, X_query: pd.DataFrame) -> tuple[dict, int]:
        num_context_rows = len(X_context)
        dummy_target = pd.Series(['a'] * num_context_rows, index=X_context.index, name='__dummy_target__')

        data, _ = self.tokenizer(
            X_context=X_context, y_context=dummy_target, X_query=X_query, task_type=TaskType.CLASSIFICATION
        )

        # Override target: set ALL rows (context + query) to -100 so they all get the
        # same mask embedding (index 0), matching context-fm's static target embedding.
        data['target'][:] = -100

        sample = {
            'data': data,
            'num_features': X_context.shape[1],
            'num_context_rows': num_context_rows,
        }
        batch = collate_stack([sample])
        return to_device(batch, self.device, raise_on_unexpected=False), num_context_rows

    def predict(self, X: pd.DataFrame, n_clusters: int) -> np.ndarray:
        from sklearn.cluster import KMeans

        embeddings = self.transform(X)
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(embeddings)


class FlexTabADModel(BaseEstimator, ABC):
    """Base class for FlexTabAD anomaly detectors.
    Handles model loading, tokenization, and inference.
    Subclasses implement _prepare_batch() to set target labels per their detection mode.
    """

    def __init__(
        self,
        checkpoint: str = 'flextab_ad_v1_0_0.pt',
        outlier_threshold: float = 0.0,
        sentence_embedder_batch_size: int = 512,
        max_context_size: int = 8192,
        max_columns: int = 250,
        max_query_size: int = 1024,
        seed: int = 42,
    ):
        self.checkpoint = hf_hub_download(repo_id=HF_REPO_ID, filename=checkpoint)
        self.outlier_threshold = outlier_threshold
        self.sentence_embedder_batch_size = sentence_embedder_batch_size
        self.max_context_size = max_context_size
        self.max_columns = max_columns
        self.max_query_size = max_query_size
        self.seed = seed

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_dict = torch.load(self.checkpoint, map_location=self.device, weights_only=True)

        self.model = FlexTabAD(num_row_tokens=1)
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
            self.model = self.model.to(dtype=self.dtype)
        else:
            self.dtype = torch.float32

        self.model.load_weights(state_dict, self.device)
        self.model.to(self.device).eval()

        self.tokenizer = Tokenizer(is_valid=True, sentence_embedder_batch_size=sentence_embedder_batch_size)
        self.X_train_: None | pd.DataFrame = None

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.shape[1] > self.max_columns:
            X = X.sample(n=self.max_columns, axis=1, random_state=self.seed)
        if len(X) > self.max_context_size:
            X = X.sample(n=self.max_context_size, axis=0, random_state=self.seed)
        self.X_train_ = X.reset_index(drop=True)
        return self

    def _tokenize(self, X_query: pd.DataFrame) -> tuple[dict, int]:
        X_train = self.X_train_
        num_context_rows = len(X_train)
        num_query_rows = len(X_query)
        total_rows = num_context_rows + num_query_rows
        data, _ = self.tokenizer(X_context=X_train, y_context=None, X_query=X_query, task_type=None)
        data['target'] = torch.full((total_rows,), fill_value=2, dtype=torch.long)
        return data, num_context_rows

    @abstractmethod
    def _prepare_batch(self, X_query: pd.DataFrame) -> dict:
        raise NotImplementedError()

    @torch.inference_mode()
    def predict_score(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError('fit() must be called before predict_score()')
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        X_test = X_test[self.X_train_.columns]

        if len(X_test) > self.max_query_size:
            return np.concatenate(
                [
                    self.predict_score(X_test.iloc[i : i + self.max_query_size])
                    for i in range(0, len(X_test), self.max_query_size)
                ]
            )

        batch = self._prepare_batch(X_test)
        out = self.model(
            data=batch['data'], task_type=TaskType.ANOMALY_DETECTION, num_context_rows=batch['num_context_rows']
        )
        return out[0, :, 0].cpu().float().numpy()

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return (self.predict_score(X_test) >= self.outlier_threshold).astype(int)


class FlexTabADUnsupervised(FlexTabADModel):
    """Unsupervised outlier detector using FlexTabAD.

    All rows (context and query) are treated as unlabelled (target=2).
    fit(X, y=None): memorizes X_train; y is ignored.
    predict_score(X_test): raw logit per row — higher means more anomalous.
    predict(X_test): binary labels (1=outlier) by thresholding predict_score.
    """

    def _prepare_batch(self, X_query: pd.DataFrame) -> dict:
        data, num_context_rows = self._tokenize(X_query)
        sample = {'data': data, 'num_features': self.X_train_.shape[1], 'num_context_rows': num_context_rows}
        batch = collate_stack([sample])
        return to_device(batch, self.device, raise_on_unexpected=False)


class FlexTabADOneClass(FlexTabADModel):
    """One-class outlier detector using FlexTabAD.

    All training points are assumed to be non-outliers (target=0).
    Query rows are unlabelled (target=2).
    fit(X, y=None): memorizes X_train; y is ignored.
    predict_score(X_test): raw logit per row — higher means more anomalous.
    predict(X_test): binary labels (1=outlier) by thresholding predict_score.
    """

    def _prepare_batch(self, X_query: pd.DataFrame) -> dict:
        data, num_context_rows = self._tokenize(X_query)
        data['target'][:num_context_rows] = 0  # context rows → inliers
        # query rows already set to 2 (unlabelled) by _tokenize
        sample = {'data': data, 'num_features': self.X_train_.shape[1], 'num_context_rows': num_context_rows}
        batch = collate_stack([sample])
        return to_device(batch, self.device, raise_on_unexpected=False)


class FlexTabADSemisupervised(FlexTabADModel):
    """Semi-supervised outlier detector using FlexTabAD.

    External label convention: 0=unlabelled, 1=outlier.
    Context rows receive their labels; query rows are unlabelled (target=2).
    fit(X, y): y is required and must follow the external convention.
    predict_score(X_test): raw logit per row — higher means more anomalous.
    predict(X_test): binary labels (1=outlier) by thresholding predict_score.
    """

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.shape[1] > self.max_columns:
            X = X.sample(n=self.max_columns, axis=1, random_state=self.seed)
        X = X.reset_index(drop=True)
        if len(X) > self.max_context_size:
            idx = X.sample(n=self.max_context_size, axis=0, random_state=self.seed).index
            y = y[idx]
            X = X.loc[idx].reset_index(drop=True)
        self.X_train_ = X
        # external 0=unlabelled → internal 2; external 1=outlier → internal 1
        self.y_train_ = np.where(y == 0, 2, y).astype(np.int64)
        return self

    def _prepare_batch(self, X_query: pd.DataFrame) -> dict:
        data, num_context_rows = self._tokenize(X_query)
        data['target'][:num_context_rows] = torch.tensor(self.y_train_, dtype=torch.long)
        # query rows already set to 2 (unlabelled) by _tokenize
        sample = {'data': data, 'num_features': self.X_train_.shape[1], 'num_context_rows': num_context_rows}
        batch = collate_stack([sample])
        return to_device(batch, self.device, raise_on_unexpected=False)


class FlexTabRelationalEntityClassifier(FlexTabClassifier):
    def __init__(self, checkpoint: str = 'flextab_multi_v1_0_0.pt', max_context_size: int = 8192):
        super().__init__(checkpoint=checkpoint, max_context_size=max_context_size)

    @retry_on_cuda_oom(default_size=1024, param_name='query_chunk_size')
    @torch.inference_mode()
    def predict_row_tokens(self, X: pd.DataFrame, query_chunk_size: int = 1024) -> torch.Tensor:
        """Predict row tokens (embeddings) for the given DataFrame."""
        check_is_fitted(self)
        X = self.ensure_dataframe(X)
        return self._predict_on_chunks(X, 0, query_chunk_size, self._predict_encoder_only)

    def _predict_encoder_only(self, tokenized_data) -> torch.Tensor:
        """Run encoder only, return query-row embeddings."""
        tokenized_data['return_encoder_output'] = True
        return self.model(**tokenized_data).squeeze(0)  # (num_query, row_tokens, hidden)

    def get_column_empty_data(self, total_length: int) -> dict:
        data = {
            'text_embeddings': torch.zeros((total_length, 1, self.tokenizer.embedding_dim), dtype=torch.float16),
            'number_percentile_floor': torch.full((total_length, 1), fill_value=-100, dtype=torch.int64),
            'number_percentile_delta': torch.zeros((total_length, 1), dtype=torch.float32),
            'date_year_month_day_weekday': torch.zeros((total_length, 1, 4), dtype=torch.int64),
            'column_embeddings': torch.zeros((1, self.tokenizer.embedding_dim), dtype=torch.float16),
        }
        return data

    @torch.inference_mode()
    def predict_on_multi_table(
        self,
        y_context: pd.Series,
        encoder_outputs_train: torch.Tensor,
        encoder_outputs_test: torch.Tensor,
        query_chunk_size: int = 1024,
    ) -> np.ndarray:
        """Run the decoder on pre-computed encoder outputs to get class probabilities."""
        assert y_context.shape[0] == encoder_outputs_train.shape[0]
        classes = unique_labels(y_context)
        all_model_probs = []

        for start_idx in trange(0, len(encoder_outputs_test), query_chunk_size):
            enc_idx = min(start_idx + query_chunk_size, len(encoder_outputs_test))
            num_query = enc_idx - start_idx
            total_length = len(y_context) + num_query

            # Build minimal target-only data (1 column, no features)
            data = self.get_column_empty_data(total_length)
            data, label_classes = self.tokenizer.process_target(data, y_context, TaskType.CLASSIFICATION)
            label_classes = label_classes[y_context.name][np.newaxis, :]

            # Add batch dimension and send to device
            data_batched = {k: v.unsqueeze(0) for k, v in data.items()}
            data_batched = to_device(data_batched, self.device, raise_on_unexpected=False)

            # concatenate context encoder outputs with current query chunk of test encoder outputs
            encoder_outputs = torch.cat([encoder_outputs_train, encoder_outputs_test[start_idx:enc_idx]], dim=0)
            encoder_outputs = encoder_outputs.unsqueeze(0).to(self.device)

            logits = self.model.forward_decoder_only(
                data_batched, TaskType.CLASSIFICATION, encoder_outputs, num_context_rows=len(y_context)
            )
            logits = logits.squeeze(0)  # (num_query, num_head_outputs)

            _, unordered_preds = self.model.extract_prediction_classification(logits, label_classes)

            test_logits = self.reorder_logits(unordered_preds, label_classes, classes)
            test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
            all_model_probs.append(test_probs)

        return np.concatenate(all_model_probs, axis=0)

    def get_tables_embeddings(
        self,
        train_subgraph_features: dict[str, pd.DataFrame],
        test_subgraph_features: dict[str, pd.DataFrame],
        entity_table_name: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Predict embeddings (row tokens) for both train and test tables."""
        train_embeddings, test_embeddings = {}, {}
        ordered = [entity_table_name] + [t for t in train_subgraph_features if t != entity_table_name]

        for table_name in ordered:
            print('Embedding table:', table_name)
            train_df = train_subgraph_features[table_name]
            dummy_target = pd.Series([0] * len(train_df), index=train_df.index, name='target')
            self.fit(train_df, dummy_target)

            train_embeddings[table_name] = self.predict_row_tokens(train_df)
            if table_name in test_subgraph_features:
                test_embeddings[table_name] = self.predict_row_tokens(test_subgraph_features[table_name])

        return train_embeddings, test_embeddings

    def gather_entity_embeddings(
        self,
        df: pd.DataFrame,
        subgraph: dict[str, pd.DataFrame],
        entity_to_rows: dict[str, dict[tuple, list]],
        entity_col: str,
        time_col: str,
        table_embeddings: dict[str, torch.Tensor],
        pkey_cols: dict[str, str],
    ) -> list[torch.Tensor]:
        """For each row in df, collect embeddings from all tables reached by BFS.

        Returns a list of length N (one per df row), where each element is a
        tensor of shape (K, 1, emb_dim) - K is the total number of
        rows reached across all tables for that entity, with no padding yet.

        Args:
            df               : entity/task DataFrame with entity_col and time_col
            subgraph         : {table_name: DataFrame} with pkey column intact
            entity_to_rows   : {table_name: {(entity_id, timestamp): [pkeys]}}
            entity_col       : name of the entity id column in df
            time_col         : name of the timestamp column in df
            table_embeddings : {table_name: (N_rows, 1, emb_dim) tensor}
            pkey_cols        : {table_name: pkey_column_name} mapping
        """
        # Build pkey -> position lookup for each table
        pkey_to_pos = {}
        for table_name in table_embeddings:
            pkey_col_name = pkey_cols[table_name]
            pkey_to_pos[table_name] = {
                int(pk): i for i, pk in enumerate(subgraph[table_name][pkey_col_name].astype(int).values)
            }

        result = []
        for eid, ts in zip(df[entity_col].values, df[time_col].values):
            entity_embeddings = []
            ts = pd.Timestamp(ts)
            for table_name, embeddings in table_embeddings.items():
                mapping = entity_to_rows.get(table_name, {})
                reached_pkeys = mapping.get((int(eid), ts), [])
                positions = [pkey_to_pos[table_name][p] for p in reached_pkeys if p in pkey_to_pos[table_name]]
                if positions:
                    entity_embeddings.append(embeddings[positions])  # (k, num_features, emb_dim)
            result.append(torch.cat(entity_embeddings, dim=0))

        return result

    def pad_entity_embeddings(self, entity_embs: list[torch.Tensor], max_rows_per_entity: int = 256) -> torch.Tensor:
        """Pad per-entity embeddings to a fixed-size tensor.

        Returns shape (N, max_rows_per_entity * num_features, emb_dim).
        """
        num_features, emb_dim = entity_embs[0].shape[1], entity_embs[0].shape[2]
        out = torch.zeros((len(entity_embs), max_rows_per_entity, num_features, emb_dim), dtype=entity_embs[0].dtype)
        for i, emb in enumerate(entity_embs):
            n = min(len(emb), max_rows_per_entity)
            out[i, :n] = emb[:n]
        return out.reshape(-1, max_rows_per_entity * num_features, emb_dim)

    def predict_proba(self, data: RelbenchData) -> np.ndarray:
        """End-to-end prediction: encode tables, gather entity embeddings, decode.

        Args:
            data: RelbenchData dataclass from prepare_relbench_dataset()

        Returns:
            Class probability matrix of shape (n_test, n_classes).
        """
        print('\nCall Encoder to get tables embeddings')
        train_embeddings, test_embeddings = self.get_tables_embeddings(
            data.train_features, data.test_features, data.entity_table
        )

        print('\nGather embeddings for each entity row and call Decoder to get predictions')
        train_entity_embs = self.gather_entity_embeddings(
            data.train_df,
            data.train_subgraph,
            data.train_entity_to_rows,
            data.entity_col,
            data.time_col,
            train_embeddings,
            data.pkey_cols,
        )
        test_entity_embs = self.gather_entity_embeddings(
            data.test_df,
            data.test_subgraph,
            data.test_entity_to_rows,
            data.entity_col,
            data.time_col,
            test_embeddings,
            data.pkey_cols,
        )

        train_entity_embs = self.pad_entity_embeddings(train_entity_embs)
        test_entity_embs = self.pad_entity_embeddings(test_entity_embs)

        target_column = data.train_df.columns[-1]
        return self.predict_on_multi_table(data.train_df[target_column], train_entity_embs, test_entity_embs)


class FlexTabMatcher(FlexTabClassifier):
    """Dual-table entity matching with FlexTab.

    Two aligned tables (X_left, X_right) describe the same set of row pairs;
    the binary target y indicates whether pairs are considered matches.
    The encoder runs over each table separately, the per-side row
    tokens are concatenated along the column axis, and the decoder
    cross-attends over both tables to produce the match prediction.

    Use ``fit_matching`` and ``predict_matching`` rather than ``fit``/``predict``.
    """

    def __init__(
        self,
        checkpoint: str = 'flextab_matching_v1_0_0.pt',
        bagging: int = 8,
        max_context_size: int = 8192,
        is_drop_constant_columns: bool = True,
        sentence_embedder_batch_size: int = 512,
        **kwargs,
    ):
        super().__init__(
            checkpoint=checkpoint,
            bagging=bagging,
            max_context_size=max_context_size,
            is_drop_constant_columns=is_drop_constant_columns,
            sentence_embedder_batch_size=sentence_embedder_batch_size,
            **kwargs,
        )

    def fit(self, X_left: pd.DataFrame, X_right: pd.DataFrame, y: pd.Series):
        """Store left/right context tables and binary match labels.

        The width/schema of the two tables do not need to match, but the number of rows must.

        Arguments:
            X_left: DataFrame with left-side context tables.
            X_right: DataFrame with right-side context tables.
            y: Binary Series indicating whether each row pair is a match.
        """
        assert len(X_left) == len(X_right), 'X_left, X_right must have the same length'
        assert len(y) == len(X_left), 'y, X_left must have the same length'

        if not isinstance(X_left, pd.DataFrame):
            X_left = pd.DataFrame(X_left)
        if not isinstance(X_right, pd.DataFrame):
            X_right = pd.DataFrame(X_right)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X_left.index, name='match')
        elif y.name is None:
            y = y.rename('match')

        if len(X_left) != len(X_right) or len(X_left) != len(y):
            raise FlexTabInputValidationError('X_left, X_right, and y must have the same length')

        X_left = X_left.reset_index(drop=True)
        X_right = X_right.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Drop rows where y is missing on both sides together so the contexts stay aligned.
        missing_mask = y.isnull()
        if missing_mask.any():
            X_left = X_left[~missing_mask].reset_index(drop=True)
            X_right = X_right[~missing_mask].reset_index(drop=True)
            y = y[~missing_mask].reset_index(drop=True)

        X_left = self.deduplicate_columns(X_left)
        X_right = self.deduplicate_columns(X_right)

        self._X_left = X_left
        self._X_right = X_right
        self.y_ = y

        # Populate self.X_, self.y_, self.classes_, self.task_type, self.target_name, self.bagging_config.
        # super().fit(X_left, y)
        self.bagging_config = self.bagging
        if X_left.shape[0] < self.max_context_size:
            self.bagging_config = 1

        self.task_type = TaskType.CLASSIFICATION
        self.target_name = y.name if y.name is not None else 'TARGET'
        self.classes_ = unique_labels(y)

        return self

    def get_one_tokenized_matching_data(self, X_left_test, X_right_test, bagging_index):
        """Tokenize left and right tables independently and return a dual-table batch dict."""
        X_left_train = self._X_left
        X_right_train = self._X_right

        n_ctx = len(X_left_train)
        if self.bagging_config > 1 or n_ctx > self.max_context_size:
            sampled = X_left_train.sample(
                min(self.max_context_size, n_ctx),
                replace=False,
                random_state=self.seed + bagging_index,
            )
            idx = sampled.index
            X_left_train = X_left_train.loc[idx]
            X_right_train = X_right_train.loc[idx]

        if self.is_drop_constant_columns:
            const_left = list(X_left_train.columns[X_left_train.nunique() == 1])
            if const_left:
                X_left_train = X_left_train.drop(columns=const_left)
            const_right = list(X_right_train.columns[X_right_train.nunique() == 1])
            if const_right:
                X_right_train = X_right_train.drop(columns=const_right)

        X_left_test = X_left_test[X_left_train.columns]
        X_right_test = X_right_test[X_right_train.columns]

        y_train = self.y_.loc[X_left_train.index]

        self.tokenizer.random_seed = self.seed + bagging_index
        data_left, label_classes = self.tokenizer(X_left_train, y_train, X_left_test, TaskType.CLASSIFICATION)
        data_right, _ = self.tokenizer(X_right_train, y_train, X_right_test, TaskType.CLASSIFICATION)

        result = {
            'data_left': data_left,
            'data_right': data_right,
            'num_query_rows': X_left_test.shape[0],
            'num_context_rows': X_left_train.shape[0],
            'num_features': X_left_train.shape[1],
            'task_type': TaskType.CLASSIFICATION,
            'label_classes': {k: np.asarray(v) for k, v in label_classes.items()},
        }
        return collate_stack([result])

    def _extract_matching_subset_and_send_to_device(self, tokenized_data, start: int, query_chunk_size: int) -> dict:
        """Extract context + a chunk of queries from a dual-table tokenized batch."""
        context_size = tokenized_data['num_context_rows']
        subset = {k: v for k, v in tokenized_data.items() if k not in ('data_left', 'data_right')}
        for side in ('data_left', 'data_right'):
            subset[side] = {}
            for key, value in tokenized_data[side].items():
                if key == 'column_embeddings':
                    subset[side][key] = value
                else:
                    ctx = value[:, :context_size]
                    qry = value[:, context_size + start : context_size + start + query_chunk_size]
                    subset[side][key] = torch.cat([ctx, qry], dim=1)
        return to_device(subset, self.device, raise_on_unexpected=False)

    def _predict_matching_chunk(self, chunk) -> torch.Tensor:
        """Run the dual-table forward and return logits reordered to self.classes_."""
        logits = self.model.forward_matching(
            data_left=chunk['data_left'],
            data_right=chunk['data_right'],
            num_context_rows=chunk['num_context_rows'],
        ).squeeze(0)
        label_classes = chunk['label_classes'][self.target_name]
        _, unordered = self.model.extract_prediction_classification(logits, label_classes)
        return self.reorder_logits(unordered, label_classes, self.classes_)

    @retry_on_cuda_oom(default_size=1024, param_name='query_chunk_size')
    @torch.inference_mode()
    def predict_proba(
        self, X_left_test: pd.DataFrame, X_right_test: pd.DataFrame, query_chunk_size: int = 1024
    ) -> np.ndarray:
        """Return per-class probabilities for each test pair."""
        check_is_fitted(self)

        assert len(X_left_test) == len(X_right_test), 'Left and right tables need to have the same number of rows.'

        if not isinstance(X_left_test, pd.DataFrame):
            X_left_test = pd.DataFrame(X_left_test)
        if not isinstance(X_right_test, pd.DataFrame):
            X_right_test = pd.DataFrame(X_right_test)
        X_left_test = self.deduplicate_columns(X_left_test)
        X_right_test = self.deduplicate_columns(X_right_test)

        predictions_for_all_bags = []
        for bagging_index in range(self.bagging_config):
            tokenized = self.get_one_tokenized_matching_data(X_left_test, X_right_test, bagging_index)
            chunk_logits = []
            for i in range(0, tokenized['num_query_rows'], query_chunk_size):
                chunk = self._extract_matching_subset_and_send_to_device(tokenized, i, query_chunk_size)
                chunk_logits.append(self._predict_matching_chunk(chunk))
            predictions_for_all_bags.append(torch.cat(chunk_logits, dim=0))

        return self._aggregate_from_bags(predictions_for_all_bags)

    def predict(
        self, X_left_test: pd.DataFrame, X_right_test: pd.DataFrame, query_chunk_size: int = 1024
    ) -> np.ndarray:
        """Return predicted class labels for each test pair."""
        probas = self.predict_proba(X_left_test, X_right_test, query_chunk_size=query_chunk_size)
        predicted_indices = probas.argmax(-1)
        return self.classes_[predicted_indices]


class FlexTabEmbedder(BaseEstimator):
    """Produce row-level embeddings of a tabular DataFrame using the FlexTab encoder.

    Call ``embed(context, query=None)`` to obtain shape ``(n_context + n_query, num_row_tokens * hidden_size)``
    embeddings — context rows first, query rows after. When ``query`` is omitted the returned tensor has
    ``len(context)`` rows and contextualized fully using self-attention.
    """

    def __init__(
        self,
        checkpoint: str = 'flextab_matching_v1_0_0.pt',
        max_context_size: int = 8192,
        is_drop_constant_columns: bool = True,
        sentence_embedder_batch_size: int = 512,
        seed: int = 42,
    ):
        self.checkpoint = hf_hub_download(repo_id=HF_REPO_ID, filename=checkpoint)
        self.max_context_size = max_context_size
        self.is_drop_constant_columns = is_drop_constant_columns
        self.sentence_embedder_batch_size = sentence_embedder_batch_size
        self.seed = seed

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.checkpoint, map_location=self.device, weights_only=True)

        self.model = FlexTab()
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
            self.model = self.model.to(dtype=self.dtype)
        else:
            self.dtype = torch.float32

        self.model.load_weights(state_dict, self.device)
        self.model.to(self.device).eval()

        self.tokenizer = Tokenizer(
            random_seed=self.seed, sentence_embedder_batch_size=sentence_embedder_batch_size, is_valid=True
        )

    @retry_on_cuda_oom(default_size=1024, param_name='query_chunk_size')
    @torch.inference_mode()
    def embed(
        self,
        context: pd.DataFrame,
        query: pd.DataFrame | None = None,
        query_chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Return row-token embeddings for ``context`` rows followed by ``query`` rows.

        Args:
            context: Context table. Subsampled to ``max_context_size`` rows if larger.
            query: Optional query table. Must share columns with ``context``. When None,
                only context-row embeddings are returned (self-context).
                Context does not attend to query and query rows do not attend to each other.
                Thus, query rows do not alter context row embeddings nor change the embeddings of other query rows.
            query_chunk_size: Number of query rows processed per forward pass; reduced
                              automatically on CUDA OOM.

        Returns:
            torch.Tensor of shape ``(n_context + n_query, num_row_tokens * hidden_size)``
            on CPU and float32. Context rows come first, query rows after; when
            ``query is None`` the result has ``len(context)`` rows.
        """
        if not isinstance(context, pd.DataFrame):
            context = pd.DataFrame(context)
        context = FlexTabModel.deduplicate_columns(context.reset_index(drop=True))
        if len(context) > self.max_context_size:
            context = context.sample(n=self.max_context_size, replace=False, random_state=self.seed).reset_index(
                drop=True
            )

        if self.is_drop_constant_columns:
            constant_cols = list(context.columns[context.nunique() == 1])
            if constant_cols:
                context = context.drop(columns=constant_cols)

        if query is None:
            return self._embed_self_context(context, query_chunk_size)

        if not isinstance(query, pd.DataFrame):
            query = pd.DataFrame(query, columns=context.columns)
        query = FlexTabModel.deduplicate_columns(query.reset_index(drop=True))
        query = query[context.columns]

        # Embed context rows by chunking them as queries against themselves, then embed the
        # real queries against the fixed context. This keeps the context embedding consistent
        # across chunks and avoids re-encoding the context for every query chunk.
        context_emb = self._embed_self_context(context, query_chunk_size)
        query_emb = self._embed_against_context(context, query, query_chunk_size)
        return torch.cat([context_emb, query_emb], dim=0)

    def _tokenize(self, context: pd.DataFrame, query: pd.DataFrame) -> dict:
        """Tokenize context+query with a dummy classification target that gets fully masked.

        The dummy target is not actively used, but the code of the encoder and tokenizer currently expect a target
        to be present.
        """
        # Mirror FlexTabClusterer: the tokenizer needs a target column, but we mask all rows
        # so the target embedding contributes only the static mask vector.
        dummy_target = pd.Series(['a'] * len(context), index=context.index, name='__embed_target__')
        data, _ = self.tokenizer(
            X_context=context, y_context=dummy_target, X_query=query, task_type=TaskType.CLASSIFICATION
        )
        data['target'][:] = -100

        sample = {
            'data': data,
            'num_context_rows': len(context),
            'num_features': context.shape[1],
        }
        return collate_stack([sample])

    def _run_encoder(self, batch: dict) -> torch.Tensor:
        """Run apply_embeddings + forward_encoder; return (R, num_row_tokens * hidden) on CPU."""
        batch = to_device(batch, self.device, raise_on_unexpected=False)
        feature_embeddings, _target_embeddings = self.model.apply_embeddings(batch['data'], TaskType.CLASSIFICATION)
        encoder_outputs = self.model.forward_encoder(feature_embeddings, batch['num_context_rows'])
        # encoder_outputs: (B=1, R, num_row_tokens, hidden) -> (R, num_row_tokens * hidden)
        return encoder_outputs.squeeze(0).flatten(start_dim=1).cpu().float()

    def _embed_against_context(self, context: pd.DataFrame, query: pd.DataFrame, query_chunk_size: int) -> torch.Tensor:
        """Run the encoder over ``context + query_chunk`` repeatedly, returning only the query halves."""
        n_ctx = len(context)
        chunks = []
        for i in range(0, len(query), query_chunk_size):
            chunk = query.iloc[i : i + query_chunk_size]
            batch = self._tokenize(context, chunk)
            full = self._run_encoder(batch)  # (n_ctx + len(chunk), hidden)
            chunks.append(full[n_ctx:])
        return torch.cat(chunks, dim=0)

    def _embed_self_context(self, context: pd.DataFrame, query_chunk_size: int) -> torch.Tensor:
        """Embed every row of ``context`` by treating ``context`` as both context and query."""
        return self._embed_against_context(context, context, query_chunk_size)
