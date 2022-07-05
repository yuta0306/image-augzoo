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
        assert len(inputs) % 2 == 0
        if torch.rand(1) > self.p:
            return inputs
        device = inputs[0].device

        cut_ratio = torch.randn(1, device=device) * 0.01 + self.alpha
        h, w = inputs[0].size(-2), inputs[0].size(-1)
        ch, cw = int(h * cut_ratio), int(w * cut_ratio)

        fcy = random.randint(0, h - ch + 1)
        fcx = random.randint(0, w - cw + 1)
        tcy, tcx = fcy, fcx

        inputs_org = (input_ for input_ in inputs[::2])
        inputs_ref = (input_ for input_ in inputs[1::2])
        scales = (input_.size(-1) // inputs[0].size(-1) for input_ in inputs[::2])
        masks = [torch.zeros(input_.size(), device=device) for input_ in inputs[::2]]
        for i, scale in enumerate(scales):
            masks[i][
                ..., tcy * scale : (tcy + ch) * scale, tcx * scale : (tcx + cw) * scale
            ] = 1

        transformed = tuple(
            input_.where(mask == 0, input_ref)
            for input_, input_ref, mask in zip(inputs_org, inputs_ref, masks)
        )

        return transformed

    def apply_batch(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        bs, c, h, w = inputs[0].size()
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if self.alpha <= 0 or (probs > self.p).all():
            return inputs

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

        fcys = [random.randint(0, h - ch + 1) for ch in chs]
        fcxs = [random.randint(0, w - cw + 1) for cw in cws]

        perm = torch.randperm(bs, device=device)
        inputs_org = (input_.clone() for input_ in inputs)
        inputs_ref = (input_[perm, ...] for input_ in inputs)
        scales = (input_.size(-1) // inputs[0].size(-1) for input_ in inputs)
        masks = [torch.zeros(input_.size(), device=device) for input_ in inputs]
        for i, scale in enumerate(scales):
            for b, (fcy, fcx, ch, cw) in enumerate(zip(fcys, fcxs, chs, cws)):
                masks[i][
                    b,
                    :,
                    fcy * scale : (fcy + ch) * scale,
                    fcx * scale : (fcx + cw) * scale,
                ] = 1

        transformed = tuple(
            input_.where(mask == 0, input_ref).where(
                (probs < self.p)
                .view(-1, 1, 1, 1)
                .expand(bs, c, input_.size(-2), input_.size(-1)),
                input_,
            )
            for input_, input_ref, mask in zip(inputs_org, inputs_ref, masks)
        )

        return transformed
