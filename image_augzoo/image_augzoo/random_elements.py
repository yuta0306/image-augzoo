from typing import Tuple

import numpy as np
import torch

from image_augzoo.core.transform import MultiTransform


class RandomElements(MultiTransform):
    """
    RandomElements

    Attributes
    ----------
    p : float
    alpha : float
    """

    def __init__(
        self,
        p: float = 1.0,
        alpha: float = 0.5,
        soft_label: bool = True,
    ):
        """
        Parameters
        ----------
        p : float
        alpha : float
        """
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self.soft_label = soft_label
        super().__init__(p=p)

    def apply(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        if torch.rand(1) >= self.p:
            return inputs, kwargs
        device = inputs[0].device

        cutout = np.random.choice(
            [0.0, 1.0], size=inputs[0].size(), p=[self.alpha, 1 - self.alpha]
        )
        mask = torch.tensor(cutout, dtype=torch.float32, device=device)

        inputs_org = (input_ for input_ in inputs[::2])
        inputs_ref = (input_ for input_ in inputs[1::2])

        labels = kwargs.get("labels")
        if self.soft_label and labels is not None:
            labels = (
                self.alpha * labels[0].float() + (1 - self.alpha) * labels[1].float()
            )
            kwargs["labels"] = labels

        return (
            tuple(
                mask * input_ + (1 - mask) * input_ref
                for input_, input_ref in zip(inputs_org, inputs_ref)
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

        cutout = np.random.choice(
            [0.0, 1.0], size=(bs, c, h, w), p=[self.alpha, 1 - self.alpha]
        )
        mask = torch.tensor(cutout, dtype=torch.float32, device=device)

        perm = torch.randperm(bs)
        inputs_ref = (input_[perm] for input_ in inputs)

        labels = kwargs.get("labels")
        if self.soft_label and labels is not None:
            labels_ref = labels[perm]
            labels = self.alpha * labels.float() + (1 - self.alpha) * labels_ref.float()
            kwargs["labels"] = labels

        return (
            tuple(
                (mask * input_ + (1 - mask) * input_ref).where(
                    (probs < self.p).view(-1, 1, 1, 1).expand(bs, c, h, w),
                    input_,
                )
                for input_, input_ref in zip(inputs, inputs_ref)
            ),
            kwargs,
        )
