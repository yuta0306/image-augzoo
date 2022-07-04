from typing import Tuple

import numpy as np
import torch

from image_augzoo.core.transform import MultiTransform


class Cutout(MultiTransform):
    """
    Cutout

    Attributes
    ----------
    p : float
    alpha : float
    """

    def __init__(
        self,
        p: float = 1.0,
        alpha: float = 0.001,
    ):
        """
        Parameters
        ----------
        p : float
        alpha : float
        """
        self.alpha = alpha
        super().__init__(p=p)

    def apply(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        if self.alpha <= 0 or torch.rand(1) >= self.p:
            return inputs

        LR = inputs[0]
        cutout = np.random.choice(
            [0.0, 1.0], size=LR.size()[1:], p=[self.alpha, 1 - self.alpha]
        )
        mask = torch.tensor(cutout, dtype=torch.float32, device=LR.device)

        return tuple(
            input_ * mask if i == 0 else input_ for i, input_ in enumerate(inputs)
        )

    def apply_batch(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        bs = inputs[0].size(0)
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if self.alpha <= 0 or (probs > self.p).all():
            return inputs

        LR = inputs[0]
        h, w = inputs[0].size(-2), inputs[0].size(-1)
        cutout = np.random.choice(
            [0.0, 1.0], size=(bs, 1, h, w), p=[self.alpha, 1 - self.alpha]
        ).repeat(3, axis=1)
        mask = torch.tensor(cutout, dtype=torch.float32, device=LR.device)

        return tuple(
            input_ * mask if i == 0 else input_ for i, input_ in enumerate(inputs)
        )