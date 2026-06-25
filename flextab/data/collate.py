from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _init_batch(example: Dict[str, Any]) -> Dict[str, Any]:
    """Create the empty batch structure based on a single example."""
    batch = {key: [] for key in example}
    for data_key in ('data', 'data_left', 'data_right'):
        if data_key in example:
            batch[data_key] = {key: [] for key in example[data_key]}
    batch['label_classes'] = defaultdict(list)
    return batch


def _get_max_label_len(examples: List[Dict[str, Any]]) -> Optional[dict[str, int]]:
    """Return the maximum length of 'label_classes' across
    examples or None if the field is absent.
    """
    if 'label_classes' in examples[0].keys():
        return {k: max(len(ex['label_classes'][k]) for ex in examples) for k in examples[0]['label_classes']}
    return None


def _append_example(
    batch: Dict[str, Any],
    ex: Dict[str, Any],
    max_labels_num: Optional[dict[str, int]],
) -> None:
    """Append a single example into the partially-built batch.

    Pads 'label_classes' to the max length when necessary.
    """
    for key, value in ex.items():
        if key in ('data', 'data_left', 'data_right'):
            for d_key, d_value in value.items():
                batch[key][d_key].append(d_value)
        elif key == 'label_classes':
            assert max_labels_num is not None, 'max_labels_num should not be None when label_classes are present'
            for d_key, d_value in value.items():
                pad_width = (0, max_labels_num[d_key] - len(d_value))
                d_value = np.pad(d_value, pad_width, mode='constant', constant_values='-100')
                batch[key][d_key].append(d_value)
        else:
            batch[key].append(value)


def _stack_or_simplify(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Stack tensors/ndarrays and simplify singleton non-array fields.

    Keeps the original behavior: data tensors are stacked, torch tensors
    and numpy arrays are stacked, and other fields asserted identical
    across examples and reduced to a single value.
    """
    for key in batch.keys():
        if key in ('data', 'data_left', 'data_right'):
            for d_key, d_value in batch[key].items():
                batch[key][d_key] = torch.stack(d_value)
        elif key == 'label_classes':
            for target_column_name, label_classes in batch['label_classes'].items():
                batch['label_classes'][target_column_name] = np.stack(label_classes)
        elif isinstance(batch[key][0], torch.Tensor):
            batch[key] = torch.stack(batch[key])
        elif isinstance(batch[key][0], np.ndarray):
            batch[key] = np.stack(batch[key])
        else:
            # non-array fields must be identical across the batch ->
            # reduce to single value
            assert all(batch[key][0] == batch[key][i] for i in range(1, len(batch[key])))
            batch[key] = batch[key][0]
    return batch


def collate_stack(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate list of examples into a batch.

    Each element in the batch has the same number of queries and the same
    number of context examples so we just need to stack tensors and numpy
    arrays. The
    function has been modularized to reduce cognitive complexity.
    """
    if not examples:
        return {}

    batch = _init_batch(examples[0])
    max_labels_num = _get_max_label_len(examples)

    for ex in examples:
        _append_example(batch, ex, max_labels_num)

    batch = _stack_or_simplify(batch)

    # Ensure label_classes is a regular dict (not defaultdict)
    batch['label_classes'] = dict(batch['label_classes'])

    return batch
