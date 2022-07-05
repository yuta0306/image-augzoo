from typing import Tuple

import torch

from image_augzoo.core.transform import MultiTransform


class Mixup(MultiTransform):
    """
    Mixup
    Attributes
    ----------
    p : float
    alpha : float
    """

    def __init__(self, p: float = 1.0, alpha: float = 1.2, soft_label: bool = False):
        """
        Parameters
        ----------
        p : float
        alpha : float
        """
        self.alpha = alpha
        self.soft_label = soft_label
        super().__init__(p=p)

    def apply(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        if self.alpha <= 0 or torch.rand(1) > self.p:
            return inputs, kwargs

        dist = torch.distributions.beta.Beta(self.alpha, self.alpha)
        v = dist.sample()

        inputs_org = (input_ for input_ in inputs[::2])
        inputs_ref = (input_ for input_ in inputs[1::2])

        labels = kwargs.get("labels")
        labels_ref = kwargs.get("labels_ref")
        if self.soft_label and labels is not None and labels_ref is not None:
            labels = v * labels.float() + (1 - v) * labels_ref.float()
            kwargs["labels"] = labels

        return (
            tuple(
                v * input_ + (1 - v) * input_ref
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

        dist = torch.distributions.beta.Beta(self.alpha, self.alpha)
        v = (
            dist.sample(torch.Size((bs,)))
            .view(-1, 1, 1, 1)
            .repeat(1, c, h, w)
            .to(device)
        )

        perm = torch.randperm(bs, device=device)
        inputs_org = (input_.clone() for input_ in inputs)
        inputs_ref = (input_[perm] for input_ in inputs)

        labels = kwargs.get("labels")
        if self.soft_label and labels is not None:
            labels_ref = labels[perm]
            labels = v * labels.float() + (1 - v) * labels_ref.float()
            kwargs["labels"] = labels

        return (
            tuple(
                (v * input_ + (1 - v) * input_ref).where(
                    (probs < self.p)
                    .view(-1, 1, 1, 1)
                    .expand(bs, c, input_.size(-2), input_.size(-1)),
                    input_,
                )
                for input_, input_ref in zip(inputs_org, inputs_ref)
            ),
            kwargs,
        )
