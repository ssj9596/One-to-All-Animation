import torch
import re

def get_torch_cuda_version():
    if not torch.version.cuda:
        return (0, 0)
    match = re.match(r"(\d+)\.(\d+)", torch.version.cuda)
    if match:
        return tuple(map(int, match.groups()))
    return (0, 0)

major, minor = get_torch_cuda_version()
ckpt_kwargs = {"use_reentrant": (major > 12) or (major == 12 and minor >= 2)}
print(f"Set gradient checkpointing: ckpt_kwargs={ckpt_kwargs} (CUDA {major}.{minor})")