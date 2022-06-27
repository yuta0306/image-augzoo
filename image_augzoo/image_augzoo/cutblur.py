from typing import Tuple

import numpy as np
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
        self.p = p
        self.alpha = alpha
        self.expand = expand
        if expand:
            raise NotImplementedError("ToDo")

    def __call__(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        if self.alpha <= 0 or np.random.rand(1) >= self.p:
            return inputs
        assert len(inputs) > 2
        is_batched = inputs[0].ndim == 4
        device = inputs[0].device

        if is_batched:
            bs = inputs[0].size(0)
            perm = torch.randperm(bs)
            inputs_org = (input_.clone() for input_ in inputs)
            inputs_ref = (input_[perm, ...] for input_ in inputs)

        cut_ratio = torch.randn(1, device=device) * 0.01 + self.alpha
        h, w = inputs[0].size(-2), inputs[0].size(-1)
        ch, cw = torch.tensor(
            h * cut_ratio, dtype=torch.int16, device=device
        ), torch.tensor(w * cut_ratio, dtype=torch.int16, device=device)
        cy = torch.randint(0, h - ch + 1, device=device)
        cx = torch.randint(0, w - cw + 1, device=device)

        # apply CutBlur to inside or outside
        if torch.randn(1) > 0.5:
            inputs_org[..., cy : cy + ch, cx : cx + cw] = inputs_ref[
                ..., cy : cy + ch, cx : cx + cw
            ]
        else:
            LR_aug = HR.clone()
            LR_aug[..., cy : cy + ch, cx : cx + cw] = LR[
                ..., cy : cy + ch, cx : cx + cw
            ]
            LR = LR_aug

        if self.expand:
            return LR, HR
        LR = T.Resize(LR_size)(LR)
        return LR, HR
