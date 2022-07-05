import random
from typing import Tuple

import torch

from image_augzoo.core.transform import DualTransform


class CutBlur(DualTransform):
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
        if self.alpha <= 0 or torch.rand(1) > self.p:
            return inputs
        device = inputs[0].device

        cut_ratio = torch.randn(1, device=device) * 0.01 + self.alpha
        h, w = inputs[0].size(-2), inputs[0].size(-1)
        ch, cw = int(h * cut_ratio), int(w * cut_ratio)
        cy = random.randint(0, h - ch + 1)
        cx = random.randint(0, w - cw + 1)

        LR = inputs[0]
        HR = inputs[1]
        # apply CutBlur to inside or outside
        if torch.rand(1) > 0.5:
            mask = torch.zeros(inputs[0].size(), device=device)
            mask[..., cy : cy + ch, cx : cx + cw] = 1
        else:
            mask = torch.ones(inputs[0].size(), device=device)
            mask[..., cy : cy + ch, cx : cx + cw] = 0

        return (LR.where(mask == 0, HR), HR)

    def apply_batch(self, *inputs: torch.Tensor, **kwargs):
        bs = inputs[0].size(0)
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if self.alpha <= 0 or (probs > self.p).all():
            return inputs

        c, h, w = inputs[0].size(-3), inputs[0].size(-2), inputs[0].size(-1)
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

        LR = inputs[0]
        HR = inputs[1]
        # apply CutBlur to inside or outside
        mask = torch.zeros(LR.size(), device=device)
        for b, cy, cx, ch, cw in zip(range(bs), cys, cxs, chs, cws):
            if torch.rand(1) > 0.5:
                mask[b, :, cy : cy + ch, cx : cx + cw] = 1
            else:
                mask[b, ...] = 1
                mask[b, :, cy : cy + ch, cx : cx + cw] = 0

        return (
            LR.where(mask == 0, HR).where(
                (probs < self.p).view(-1, 1, 1, 1).expand(bs, c, h, w),
                LR,
            ),
            HR,
        )
