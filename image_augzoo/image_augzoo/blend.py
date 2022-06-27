from typing import Tuple

import numpy as np
import torch

from image_augzoo.core.transform import DualTransform


class Blend(DualTransform):
    """
    Blend

    Attributes
    ----------
    p : float
    alpha : float
    rgb_range : float
    """

    def __init__(
        self,
        p: float = 1.0,
        alpha: float = 0.6,
        rgb_range: float = 1.0,
        channel_first: bool = True,
    ):
        """
        Parameters
        ----------
        p : float
        alpha : float
        rgb_range : float

        Raises
        ------
        ValueError
            rgb_range is not 1.0 and 255.0
        """
        if rgb_range not in (1.0, 255.0):
            raise ValueError(
                f"rgb_range must be 1.0 or 255.0, but {rgb_range} were given"
            )
        self.p = p
        self.alpha = alpha
        self.rgb_range = rgb_range
        self.channel_first = channel_first

    def __call__(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        if self.alpha <= 0 or np.random.rand(1) >= self.p:
            return inputs

        is_batched = inputs[0].ndim == 4
        if is_batched:
            bs = inputs[0].size(0)
            c = torch.empty((bs, 3, 1, 1), device=inputs[0].device).uniform_(
                0, self.rgb_range
            )
            refs = (
                c.repeat((1, 1, input_.size(-2), input_.size(-1))) for input_ in inputs
            )
        else:
            c = torch.empty((3, 1, 1), device=inputs[0].device).uniform_(
                0, self.rgb_range
            )
            refs = (
                c.repeat((1, input_.size(-2), input_.size(-1))) for input_ in inputs
            )

        v = torch.empty(1).uniform_(self.alpha, 1)
        transformed = tuple(
            v * input_ + (1 - v) * ref for (input_, ref) in zip(inputs, refs)
        )

        return transformed
