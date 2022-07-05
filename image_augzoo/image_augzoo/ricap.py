from typing import Tuple

import numpy as np
import torch

from image_augzoo.core.transform import MultiTransform


class RICAP(MultiTransform):
    """
    RICAP
    Data Augmentation using Random Image Cropping and Patching for Deep CNNs
    [Arxiv](https://arxiv.org/pdf/1811.09030.pdf)

    Attributes
    ----------
    p : float
    beta: float
    """

    def __init__(self, p: float = 1.0, beta: float = 0.3):
        """
        Parameters
        ----------
        p : float
        beta : float
        """
        self.beta = beta
        super().__init__(p=p)

    def apply(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == 4
        h, w = inputs[0].size()[-2:]
        if self.beta <= 0 or torch.rand(1) > self.p:
            return inputs
        device = inputs[0].device

        boundary_h, boundary_w = np.random.beta(self.beta, self.beta, (2,))
        boundary_h, boundary_w = round(boundary_h * h), round(boundary_w * w)
        w1 = w3 = boundary_w
        w2 = w4 = w - boundary_w
        h1 = h2 = boundary_h
        h3 = h4 = h - boundary_h
        x1, y1 = round(np.random.uniform(0, w - w1)), round(
            np.random.uniform(0, h - h1)
        )
        x2, y2 = round(np.random.uniform(0, w - w2)), round(
            np.random.uniform(0, h - h2)
        )
        x3, y3 = (
            round(np.random.uniform(0, w - w3)),
            round(np.random.uniform(0, h - h3)),
        )
        x4, y4 = (
            round(np.random.uniform(0, w - w4)),
            round(np.random.uniform(0, h - h4)),
        )

        background = torch.zeros_like(inputs[0], device=device)
        background[..., :w1, :h1] = inputs[0][..., x1 : x1 + w1, y1 : y1 + h1]
        background[..., boundary_w : boundary_w + w2, :h2] = inputs[1][
            ..., x2 : x2 + w2, y2 : y2 + h2
        ]
        background[..., :w3, boundary_h : boundary_h + h3] = inputs[2][
            ..., x3 : x3 + w3, y3 : y3 + h3
        ]
        background[
            ..., boundary_w : boundary_w + w4, boundary_h : boundary_h + h4
        ] = inputs[3][..., x4 : x4 + w4, y4 : y4 + h4]

        return (background,)

    def apply_batch(self, *inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, ...]:
        bs, _, h, w = inputs[0].size()
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if (probs > self.p).all():
            return inputs

        boundary_h, boundary_w = np.random.beta(self.beta, self.beta, (2, bs))
        boundary_h, boundary_w = (boundary_h * h).astype(int), (boundary_w * w).astype(
            int
        )
        print(boundary_h, boundary_w)
        w1 = w3 = boundary_w
        w2 = w4 = w - boundary_w
        h1 = h2 = boundary_h
        h3 = h4 = h - boundary_h
        print(w1, w2, w3, w4)
        print(h1, h2, h3, h4)
        x1, y1 = np.random.uniform(0, w - w1, (bs,)).astype(int), np.random.uniform(
            0, h - h1, (bs,)
        ).astype(int)
        x2, y2 = (
            np.random.uniform(0, w - w2, (bs,)).astype(int),
            np.random.uniform(0, h - h2, (bs,)).astype(int),
        )
        x3, y3 = (
            np.random.uniform(0, w - w3, (bs,)).astype(int),
            np.random.uniform(0, h - h3, (bs,)).astype(int),
        )
        x4, y4 = (
            np.random.uniform(0, w - w4, (bs,)).astype(int),
            np.random.uniform(0, h - h4, (bs,)).astype(int),
        )
        print(x1, x2, x3, x4)
        print(y1, y2, y3, y4)

        inputs_ref1 = (
            input_[torch.arange(0, bs).repeat(2)[1 : 1 + bs]] for input_ in inputs
        )
        inputs_ref2 = (
            input_[torch.arange(0, bs).repeat(2)[2 : 2 + bs]] for input_ in inputs
        )
        inputs_ref3 = (
            input_[torch.arange(0, bs).repeat(2)[3 : 3 + bs]] for input_ in inputs
        )

        mask = torch.zeros_like(inputs[0], device=device, dtype=torch.int16)
        for b in range(bs):
            mask[b, :, : w1[b], : h1[b]] = 0
            mask[b, :, boundary_w[b] : boundary_w[b] + w2[b], : h2[b]] = 1
            mask[b, :, : w3[b], boundary_h[b] : boundary_h[b] + h3[b]] = 2
            mask[
                b,
                :,
                boundary_w[b] : boundary_w[b] + w4[b],
                boundary_h[b] : boundary_h[b] + h4[b],
            ] = 3
        return tuple(
            input_.where((mask == 0), ref1)
            .where(((mask == 0) | (mask == 1)), ref2)
            .where(((mask == 0) | (mask == 1) | (mask == 2)), ref3)
            for input_, ref1, ref2, ref3 in zip(
                inputs, inputs_ref1, inputs_ref2, inputs_ref3
            )
        )
