import collections.abc

from itertools import repeat
from diffusers.utils import is_torch_version
try:
    # flash attn3 == cuda >= 12.3 == "use_reentrant": True
    from flash_attn_interface import flash_attn_varlen_func as flash_attn3_varlen_func
    flash_attn_backend = "flash-attn-3"
    # print("[INFO] Using FlashAttention 3")
    
    ckpt_kwargs = {"use_reentrant": True} if is_torch_version(">=", "1.11.0") else {}
except ImportError:
    flash_attn_backend = "flash-attn-2"
    ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}



def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def as_tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def as_list_of_2tuple(x):
    x = as_tuple(x)
    if len(x) == 1:
        x = (x[0], x[0])
    assert len(x) % 2 == 0, f"Expect even length, got {len(x)}."
    lst = []
    for i in range(0, len(x), 2):
        lst.append((x[i], x[i + 1]))
    return lst


