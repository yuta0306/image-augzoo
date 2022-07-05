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

    def apply(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        if self.alpha <= 0 or torch.rand(1) >= self.p:
            return inputs, kwargs

        LR = inputs[0]
        cutout = np.random.choice(
            [0.0, 1.0], size=LR.size()[1:], p=[self.alpha, 1 - self.alpha]
        )
        mask = torch.tensor(cutout, dtype=torch.float32, device=LR.device)

        return (
            tuple(
                input_ * mask if i == 0 else input_ for i, input_ in enumerate(inputs)
            ),
            kwargs,
        )

    def apply_batch(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        bs, c, h, w = inputs[0].size()
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if self.alpha <= 0 or (probs > self.p).all():
            return inputs, kwargs

        LR = inputs[0]
        cutout = np.random.choice(
            [0.0, 1.0], size=(bs, 1, h, w), p=[self.alpha, 1 - self.alpha]
        ).repeat(3, axis=1)
        mask = torch.tensor(cutout, dtype=torch.float32, device=LR.device)

        return (
            tuple(
                (input_ * mask).where(
                    (probs < self.p).view(-1, 1, 1, 1).expand(bs, c, h, w),
                    input_,
                )
                if i == 0
                else input_
                for i, input_ in enumerate(inputs)
            ),
            kwargs,
        )
