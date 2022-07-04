import random
from typing import Tuple

import torch

from image_augzoo.core.transform import MultiTransform


class CutMix(MultiTransform):
    """
    CutMix

    Attributes
    ----------
    p : float
    alpha : float
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
        """
        self.alpha = alpha
        self.expand = expand
        if expand:
            raise NotImplementedError("ToDo")
        super().__init__(p=p)

    def apply(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) % 2 == 0
        if torch.rand(1) > self.p:
            return inputs
        device = inputs[0].device

        # scales = HR.size(1) // LR.size(1)
        cut_ratio = torch.randn(1, device=device) * 0.01 + self.alpha
        h, w = inputs[0].size(-2), inputs[0].size(-1)
        ch, cw = int(h * cut_ratio), int(w * cut_ratio)

        fcy = random.randint(0, h - ch + 1)
        fcx = random.randint(0, w - cw + 1)
        tcy, tcx = fcy, fcx

        inputs_org = (input_ for input_ in inputs[::2])
        inputs_ref = (input_ for input_ in inputs[1::2])
        scales = (input_.size(-1) // inputs[0].size(-1) for input_ in inputs[::2])
        masks = [
            torch.zeros(inputs[0].size(), device=device)
            for _ in range(len(inputs) // 2)
        ]
        for i, scale in enumerate(scales):
            masks[i][
                ..., tcy * scale : (tcy + ch) * scale, tcx * scale : (tcx + cw) * scale
            ] = 1

        transformed = tuple(
            input_.where(mask == 0, input_ref)
            for input_, input_ref, mask in zip(inputs_org, inputs_ref, masks)
        )

        return transformed
