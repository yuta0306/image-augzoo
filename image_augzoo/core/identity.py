from typing import Tuple

import torch
from image_augzoo.core.transform import MultiTransform


class Identity(MultiTransform):
    """
    Identity
    """

    def __init__(self):
        super().__init__()

    def apply(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        return inputs

    def apply_batch(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        return inputs
