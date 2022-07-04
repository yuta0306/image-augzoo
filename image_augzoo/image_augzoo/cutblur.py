import random
from typing import Tuple

import torch

from image_augzoo.core.transform import MultiTransform


class CutBlur(MultiTransform):
    """
    CutBlur

    Attributes
    ----------
    p : float
    alpha : float
    expand : bool
    """

    def __init__(
        self,
        p: float = 1.0,
        alpha: float = 0.7,
        expand: bool = False,
    ):
        """
        Parameters
        ----------
        p : float
        alpha : float
        expand : bool
        """
        self.alpha = alpha
        self.expand = expand
        if expand:
            raise NotImplementedError("ToDo")
        super().__init__(p=p)

    def apply(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) % 2 == 0
        if self.alpha <= 0 or torch.rand(1) > self.p:
            return inputs
        device = inputs[0].device

        cut_ratio = torch.randn(1, device=device) * 0.01 + self.alpha
        h, w = inputs[0].size(-2), inputs[0].size(-1)
        ch, cw = int(h * cut_ratio), int(w * cut_ratio)
        cy = random.randint(0, h - ch + 1)
        cx = random.randint(0, w - cw + 1)

        inputs_org = (input_ for input_ in inputs[::2])
        inputs_ref = (input_ for input_ in inputs[1::2])
        # apply CutBlur to inside or outside
        if torch.rand(1) > 0.5:
            mask = torch.zeros(inputs[0].size(), device=device)
            mask[..., cy : cy + ch, cx : cx + cw] = 1
        else:
            mask = torch.ones(inputs[0].size(), device=device)
            mask[..., cy : cy + ch, cx : cx + cw] = 0
        transformed = tuple(
            input_.where(mask == 0, input_ref)
            for input_, input_ref in zip(inputs_org, inputs_ref)
        )
        return transformed

    def apply_batch(self, *inputs: torch.Tensor, **kwargs):
        bs = inputs[0].size(0)
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if self.alpha <= 0 or (probs > self.p).all():
            return inputs

        h, w = inputs[0].size(-2), inputs[0].size(-1)
        chs = (
            (h * (torch.randn(bs, device=device) * 0.01 + self.alpha))
            .to(torch.int16)
            .tolist()
        )
        cws = (
            (w * (torch.randn(bs, device=device) * 0.01 + self.alpha))
            .to(torch.int16)
            .tolist()
        )
        cys = (random.randint(0, h - ch + 1) for ch in chs)
        cxs = (random.randint(0, w - cw + 1) for cw in cws)

        inputs_org = (input_ for input_ in inputs)
        perm = torch.randperm(bs, device=device)
        inputs_ref = (input_[perm, ...].clone() for input_ in inputs)
        # apply CutBlur to inside or outside
        mask = torch.zeros(inputs[0].size(), device=device)
        for b, cy, cx, ch, cw in zip(range(bs), cys, cxs, chs, cws):
            if torch.rand(1) > 0.5:
                mask[b, :, cy : cy + ch, cx : cx + cw] = 1
            else:
                mask[b, ...] = 1
                mask[b, :, cy : cy + ch, cx : cx + cw] = 0
        transformed = tuple(
            input_.where(mask == 0, input_ref).where(
                (probs < self.p)
                .view(-1, 1, 1, 1)
                .expand(bs, input_.size(1), input_.size(2), input_.size(3)),
                input_,
            )
            for input_, input_ref in zip(inputs_org, inputs_ref)
        )
        return transformed
