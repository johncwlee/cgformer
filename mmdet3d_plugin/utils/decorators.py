# Minimal local replacement for mmcv.runner.force_fp32
import functools, inspect
import torch
import torch.nn as nn
from torch.amp import custom_bwd as _custom_bwd, custom_fwd as _custom_fwd

def _cast_dtype(x, dtype):
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return x.to(dtype)
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_cast_dtype(v, dtype) for v in x)
    if isinstance(x, dict):
        return {k: _cast_dtype(v, dtype) for k, v in x.items()}
    return x

def _cast_to_fp32(x):
    return _cast_dtype(x, torch.float32)

def _cast_to_fp16(x):
    return _cast_dtype(x, torch.float16)

def force_fp32(apply_to=None):
    def decorator(old_func):
        @functools.wraps(old_func)
        def wrapper(self, *args, **kwargs):
            spec = inspect.getfullargspec(old_func)
            arg_names = spec.args[1:1+len(args)]  # skip self
            to_cast = set(spec.args[1:] if apply_to is None else apply_to)
            new_args = list(args)
            for i, name in enumerate(arg_names):
                if name in to_cast:
                    new_args[i] = _cast_to_fp32(new_args[i])
            for k in list(kwargs.keys()):
                if k in to_cast:
                    kwargs[k] = _cast_to_fp32(kwargs[k])
            return old_func(self, *new_args, **kwargs)
        return wrapper
    return decorator


def auto_fp16(apply_to=None, out_fp32=False, device_type='cuda', amp_dtype=torch.float16):
    """
    Replacement for MMCV's @auto_fp16:
      - Enters a torch.autocast context (AMP) for forward math
      - Optionally casts specified arguments to fp16 before the call (apply_to)
      - Optionally casts outputs back to fp32 (out_fp32)
    """
    def decorator(old_func):
        @functools.wraps(old_func)
        def wrapper(self, *args, **kwargs):
            # Keep legacy behavior: only active if module opted into fp16
            if not isinstance(self, nn.Module) or not getattr(self, 'fp16_enabled', False):
                return old_func(self, *args, **kwargs)

            # Selective pre-cast (optional; for memory/perf) like original apply_to
            if apply_to is not None:
                spec = inspect.getfullargspec(old_func)
                arg_names = spec.args[1:1+len(args)]  # skip self
                new_args = list(args)
                to_cast = set(apply_to)
                for i, name in enumerate(arg_names):
                    if name in to_cast:
                        new_args[i] = _cast_to_fp16(new_args[i])
                for k in list(kwargs.keys()):
                    if k in to_cast:
                        kwargs[k] = _cast_to_fp16(kwargs[k])
                args = tuple(new_args)

            # AMP region (recommended modern backend)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=True):
                out = old_func(self, *args, **kwargs)

            if out_fp32:
                out = _cast_to_fp32(out)
            return out
        return wrapper
    return decorator


def custom_bwd(*args, **kwargs):
    return _custom_bwd(*args, device_type='cuda', **kwargs)
def custom_fwd(*args, **kwargs):
    return _custom_fwd(*args, device_type='cuda', **kwargs)