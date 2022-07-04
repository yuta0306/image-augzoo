from typing import List, Optional, Tuple

import numpy as np
import torch
from image_augzoo.core.transform import MultiTransform, Transform


class Oneof(MultiTransform):
    """
    OneOf

    Attributes
    ----------
    augs : List[Transform]
    probs : Optional[List[float]]
    """

    def __init__(
        self,
        augs: List[Transform],
        probs: Optional[List[float]] = None,
    ):
        """
        Parameters
        ----------
        augs : List[Transform]
        probs : optional, List[float]
        """
        self.augs = augs
        self.probs = probs
        super().__init__()

    def apply(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        idx = np.random.choice(len(self.augs), p=self.probs)
        aug = self.augs[idx]

        return aug(*inputs, **kwargs)

    def apply_batch(self, *inputs: torch.Tensor, **kwargs):
        idx = np.random.choice(len(self.augs), p=self.probs)
        aug = self.augs[idx]

        return aug(*inputs, **kwargs)
