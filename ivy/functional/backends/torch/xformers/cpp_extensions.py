import xformers
import torch
from typing import List


def matmul_with_mask(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> List[torch.Tensor]:
    return xformers.torch.ops.matmul_with_mask(a, b, mask)
