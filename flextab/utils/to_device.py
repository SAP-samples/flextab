from typing import Optional, Union

import torch


def to_device(x, device: Union[torch.device, int], dtype: Optional[torch.dtype] = None, raise_on_unexpected=True):
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            target_dtype = dtype if v.dtype == torch.float32 else v.dtype
            x[k] = v.to(device, dtype=target_dtype)
        elif isinstance(v, dict):
            if k == 'label_classes':
                x[k] = v
            else:
                x[k] = to_device(v, device, dtype=dtype)
        elif isinstance(v, int):
            assert k in [
                'num_targets',
                'num_query_rows',
                'num_context_rows',
                'num_features',
                'train_size',
                'fold_idx',
                'is_valid',
            ]
            x[k] = v
        elif isinstance(v, list):
            assert k == 'task_types'
            x[k] = v
        elif v is not None and raise_on_unexpected:
            raise ValueError(f'Unknown type, {type(v)}')
    return x
