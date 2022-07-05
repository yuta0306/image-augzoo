from typing import Tuple

import torch

from image_augzoo.core.transform import MultiTransform


class RGBPermutation(MultiTransform):
    """
    RGBPermutation

    Attributes
    ----------
    p : float
    """

    def __init__(self, p: float = 1.0):
        """
        Parameters
        ----------
        p : float
        """
        super().__init__(p=p)

    def apply(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        if torch.rand(1) > self.p:
            return inputs
        device = inputs[0].device

        perm = torch.randperm(3, device=device)

        return tuple(input_[perm] for input_ in inputs)

    def apply_batch(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        bs, _, h, w = inputs[0].size()
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if (probs > self.p).all():
            return inputs

        perm = torch.cat([torch.randperm(3, device=device) for _ in range(bs)])
        # bias
        bias = torch.linspace(0, bs, bs * 3, device=device).to(torch.int16)
        bias[-1] = bs - 1
        bias = bias * 3
        perm = perm + bias

        return tuple(
            input_.view(-1, h, w)[perm]
            .view(bs, 3, h, w)
            .where(
                (probs < self.p)
                .view(-1, 1, 1, 1)
                .expand(bs, 3, input_.size(-2), input_.size(-1)),
                input_,
            )
            for input_ in inputs
        )
