from typing import Tuple

import numpy as np
import torch

from image_augzoo.core.transform import MultiTransform


class CutMixup(MultiTransform):
    """
    CutMixup
    CutMix & Mixup

    Attributes
    ----------
    mixup_p : float
    cutmix_p : float
    mixup_alpha : float
    cutmix_alpha : float
    """

    def __init__(
        self,
        p: float = 1.0,
        mixup_alpha: float = 1.2,
        cutmix_alpha: float = 0.7,
    ):
        """
        Parameters
        ----------
        mixup_p : float
        cutmix_p : float
        mixup_alpha : float
        cutmix_alpha : float
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        super().__init__(p=p)

    def apply(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        assert len(inputs) % 2 == 0
        if torch.rand(1) > self.p:
            return inputs, kwargs
        device = inputs[0].device

        # mixup
        dist = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha)
        v = dist.sample()
        inputs_org = (
            v * input_ + (1 - v) * input_ref
            for input_, input_ref in zip(inputs[::2], inputs[1::2])
        )
        inputs_ref = (input_ for input_ in inputs[1::2])

        # cutmix
        cut_ratio = torch.randn(1, device=device) * 0.01 + self.cutmix_alpha
        h, w = inputs[0].size(-2), inputs[0].size(-1)
        ch, cw = int(h * cut_ratio), int(w * cut_ratio)
        fcy = np.random.randint(0, h - ch + 1)
        fcx = np.random.randint(0, w - cw + 1)
        tcy, tcx = fcy, fcx
        scales = (input_.size(-1) // inputs[0].size(-1) for input_ in inputs[::2])

        # apply mixup to inside or outside
        if np.random.random() > 0.5:
            masks = [
                torch.zeros(input_.size(), device=device) for input_ in inputs[::2]
            ]
            for i, scale in enumerate(scales):
                masks[i][
                    ...,
                    tcy * scale : (tcy + ch) * scale,
                    tcx * scale : (tcx + cw) * scale,
                ] = 1
        else:
            masks = [torch.ones(input_.size(), device=device) for input_ in inputs[::2]]
            for i, scale in enumerate(scales):
                masks[i][
                    ...,
                    tcy * scale : (tcy + ch) * scale,
                    tcx * scale : (tcx + cw) * scale,
                ] = 0

        return (
            tuple(
                input_.where(mask == 0, input_ref)
                for input_, input_ref, mask in zip(inputs_org, inputs_ref, masks)
            ),
            kwargs,
        )

    def apply_batch(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        bs, c, h, w = inputs[0].size()
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if (probs > self.p).all():
            return inputs, kwargs

        # mixup
        dist = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha)
        v = (
            dist.sample(torch.Size((bs,)))
            .view(-1, 1, 1, 1)
            .repeat(1, c, h, w)
            .to(device)
        )

        perm = torch.randperm(bs, device=device)
        inputs_org = (
            v * input_ + (1 - v) * input_ref
            for input_, input_ref in zip(
                inputs, (input_[perm, ...] for input_ in inputs)
            )
        )
        inputs_ref = (input_[perm, ...] for input_ in inputs)

        # cutmix
        chs = (
            (h * (torch.randn(bs, device=device) * 0.01 + self.cutmix_alpha))
            .to(torch.int16)
            .tolist()
        )
        cws = (
            (w * (torch.randn(bs, device=device) * 0.01 + self.cutmix_alpha))
            .to(torch.int16)
            .tolist()
        )

        fcys = [np.random.randint(0, h - ch + 1) for ch in chs]
        fcxs = [np.random.randint(0, w - cw + 1) for cw in cws]
        scales = (input_.size(-1) // inputs[0].size(-1) for input_ in inputs)
        masks = [torch.zeros(input_.size(), device=device) for input_ in inputs]

        # apply mixup to inside or outside
        to_inside = [torch.rand(1) for _ in range(bs)]
        for i, scale in enumerate(scales):
            for b, (fcy, fcx, ch, cw, p) in enumerate(
                zip(fcys, fcxs, chs, cws, to_inside)
            ):
                if p > 0.5:
                    masks[i][
                        b,
                        :,
                        fcy * scale : (fcy + ch) * scale,
                        fcx * scale : (fcx + cw) * scale,
                    ] = 1
                else:
                    masks[i][b] = torch.ones_like(masks[i][b])
                    masks[i][
                        b,
                        :,
                        fcy * scale : (fcy + ch) * scale,
                        fcx * scale : (fcx + cw) * scale,
                    ] = 0

        return (
            tuple(
                input_.where(mask == 0, input_ref).where(
                    (probs < self.p)
                    .view(-1, 1, 1, 1)
                    .expand(bs, c, input_.size(-2), input_.size(-1)),
                    input_,
                )
                for input_, input_ref, mask in zip(inputs_org, inputs_ref, masks)
            ),
            kwargs,
        )
