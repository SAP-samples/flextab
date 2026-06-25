import os
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from flextab.constants import text_embedding_model_properties
from flextab.model.gpu_utils import retry_on_cuda_oom


class SentenceEmbedder:
    def __init__(self, sentence_embedding_model_name, batch_size=256, device=None):
        super().__init__()
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.model = AutoModel.from_pretrained(sentence_embedding_model_name)
        self.embedding_dimension = text_embedding_model_properties[sentence_embedding_model_name]['dimension']
        self.pooling_method = text_embedding_model_properties[sentence_embedding_model_name]['pooling']
        self.batch_size = batch_size
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_embedding_model_name, use_fast=True)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = self.model.to(self.device).eval()
        if torch.cuda.is_available():
            self.model = self.model.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def pooling(self, model_output, attention_mask):
        """model_output[1] claims to contain the "pooled value", which would
        already have the correct shape, but actually the manual says _not_
        to use it, and instead do the either average or first token [CLS]
        pooling, depending on the model.
        Either case, returns a tensor of shape (batch_size, embedding_dim)
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        if self.pooling_method == 'mean':
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).type(token_embeddings.dtype)
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        assert self.pooling_method == 'cls'
        return token_embeddings[:, 0].type(token_embeddings.dtype)

    @torch.no_grad()
    def embed_sentences(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """This method processes tokenized inputs in batches to generate sentence embeddings.

        Returns:
            torch.Tensor: Sentence embeddings of shape (num_texts, embedding_dim) in float16 format.
        """
        results = []
        for start_idx in range(0, len(input_ids), self.batch_size):
            these_ids = input_ids[start_idx : start_idx + self.batch_size].to(self.device)
            this_mask = attention_mask[start_idx : start_idx + self.batch_size].to(self.device)
            model_output = self.model(these_ids, this_mask)
            results.append(self.pooling(model_output, this_mask))
        res = torch.concat(results)
        if res.dtype != torch.float16:
            res = res.type(torch.float16)
        return res

    @retry_on_cuda_oom(default_size=256, param_name='batch_size')
    def _embed_batch(self, sorted_input_ids_list: list, start: int, batch_size: int) -> tuple[torch.Tensor, int, int]:
        """Process a single batch of inputs with OOM retry via decorator.

        Returns:
            Tuple of (embeddings tensor, number of items processed, effective batch_size used).
        """
        end = min(start + batch_size, len(sorted_input_ids_list))
        batch_ids_list = sorted_input_ids_list[start:end]
        if not batch_ids_list:
            return torch.tensor([], dtype=torch.float16, device=self.device), 0, batch_size
        batch_max_len = max(len(ids) for ids in batch_ids_list)

        # Pad this batch to its own max length
        padded = np.zeros((len(batch_ids_list), batch_max_len), dtype=np.int64)
        for i, ids in enumerate(batch_ids_list):
            padded[i, : len(ids)] = ids

        input_ids = torch.tensor(padded, device=self.device)
        attention_mask = (input_ids != 0).long()
        embeddings = self.embed_sentences(input_ids, attention_mask)
        return embeddings, len(batch_ids_list), batch_size

    def embed(self, texts_or_input_ids: List[str | list[int]], batch_size: int = None) -> torch.Tensor:
        """Tokenize, sort by length, batch, embed, and restore original order.
        OOM recovery handled by @retry_on_cuda_oom decorator on _embed_batch.

        Args:
            texts_or_input_ids (List[str | list[int]]): List of strings to tokenize or List of lists with token ids
            batch_size (int, optional): Batch size for processing. Defaults to self.batch_size.

        Returns:
            torch.Tensor: Sentence embeddings of shape (len(texts_or_input_ids), embedding_dim) in float16.
        """
        if not len(texts_or_input_ids):
            return torch.tensor([], dtype=torch.float16)

        if batch_size is None:
            batch_size = self.batch_size

        if isinstance(texts_or_input_ids[0], str):
            encoded = self.tokenizer.batch_encode_plus(
                texts_or_input_ids,
                padding=False,
                truncation=True,
                max_length=512,
                return_tensors=None,
            )
            input_ids_list = encoded['input_ids']
        else:
            input_ids_list = texts_or_input_ids

        num_samples = len(input_ids_list)

        # Sort by sequence length (ascending) to make batches more uniform
        lengths = [len(ids) for ids in input_ids_list]
        sorted_idx = sorted(range(num_samples), key=lambda i: lengths[i])
        sorted_input_ids_list = [input_ids_list[i] for i in sorted_idx]

        # Embed in batches (OOM recovery via decorator)
        batch_outputs = []
        start = 0
        current_batch_size = batch_size
        while start < num_samples:
            batch_embs, num_processed, current_batch_size = self._embed_batch(
                sorted_input_ids_list, start, batch_size=current_batch_size
            )
            if num_processed == 0:
                break  # Prevent infinite loop on empty batches
            batch_outputs.append(batch_embs)
            start += num_processed

        # Concatenate embeddings in sorted order
        embeddings_sorted = torch.cat(batch_outputs, dim=0)

        # Restore original order
        inv_perm = torch.empty(num_samples, dtype=torch.long, device=embeddings_sorted.device)
        inv_perm[sorted_idx] = torch.arange(num_samples, device=embeddings_sorted.device)
        return embeddings_sorted[inv_perm]

    def embed_to_bytes(self, texts: List[str]):
        """Tokenize, embed, and serialize embeddings to bytes format.

        Args:
            texts (List[str]): List of text strings to embed and serialize.

        Returns:
            List[bytes]: List of byte strings, one per input text where each bytes object is
            a serialized embedding.
        """
        if not len(texts):
            return []
        embeddings = self.embed(texts)
        embeddings = embeddings.cpu().numpy()
        if self.dtype != torch.float16:
            embeddings = embeddings.astype('float16')
        return [embedding.tobytes() for embedding in embeddings]
