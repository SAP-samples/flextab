import functools
import gc
import logging
import traceback

import torch

logger = logging.getLogger(__name__)


def retry_on_cuda_oom(default_size: int, *, param_name: str = 'query_chunk_size', min_size: int = 1):
    """Decorator to retry PyTorch code when a CUDA OOM error happens.
    Reduces the value of size parameter (param_name) on each retry until it reaches min_size.

    Args:
        default_size: Default value for the size parameter.
        param_name: Name of the kwarg to reduce (default: 'query_chunk_size').
        min_size: Minimum size before giving up (default: 1).
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_size = kwargs.get(param_name, default_size)
            kwargs = kwargs.copy()
            while current_size >= min_size:
                try:
                    kwargs[param_name] = current_size
                    return func(*args, **kwargs)
                except torch.OutOfMemoryError as e:
                    msg = traceback.format_exc().lower()
                    logger.error(f'Caught torch.OutOfMemoryError: {e}\n{msg}')
                    if current_size > min_size:
                        old_size = current_size
                        current_size = current_size // 2
                        if current_size < min_size:
                            current_size = min_size
                        logger.warning(
                            f'CUDA OOM caught with {param_name}={old_size}. Retrying with {param_name}={current_size}.'
                        )
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        raise e

        return wrapper

    return decorator
