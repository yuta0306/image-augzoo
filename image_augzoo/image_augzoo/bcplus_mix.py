from typing import Tuple

import torch

from image_augzoo.core.transform import MultiTransform


class BCPlusMix(MultiTransform):
    """
    BCPlusMix
    Attributes
    ----------
    p : float
    soft_label : bool
    """

    def __init__(self, p: float = 1.0, soft_label: bool = True):
        """
        Parameters
        ----------
        p : float
        soft_label : float, default=True
        """
        self.soft_label = soft_label
        super().__init__(p=p)

    def apply(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        if torch.rand(1) > self.p:
            return inputs, kwargs

        dist = torch.distributions.uniform.Uniform(0, 1)
        v = dist.sample()

        inputs_org = (input_ for input_ in inputs[::2])
        inputs_ref = (input_ for input_ in inputs[1::2])
        stats = (torch.std_mean(input_) for input_ in inputs[::2])  # (std, mean)
        stats_ref = (torch.std_mean(input_) for input_ in inputs[1::2])
        ps = (
            1.0 / (1.0 + (std0 / std1) * ((1.0 - v) / v))
            for (std0, _), (std1, _) in zip(stats, stats_ref)
        )

        labels = kwargs.get("labels")
        if self.soft_label and labels is not None:
            labels = v * labels[0].float() + (1 - v) * labels[1].float()
            kwargs["labels"] = labels

        return (
            tuple(
                (
                    (p * input_ + (1.0 - p) * input_ref)
                    / torch.sqrt(torch.pow(p, 2) + torch.pow(1 - p, 2))
                ).clamp(0.0, 1.0)
                for input_, input_ref, p in zip(inputs_org, inputs_ref, ps)
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

        dist = torch.distributions.uniform.Uniform(0, 1)
        v = dist.sample((bs, 1))

        perm = torch.randperm(bs)
        inputs_org = (input_.clone() for input_ in inputs)
        inputs_ref = (input_[perm] for input_ in inputs)
        stats = (torch.std_mean(input_, dim=(1, 2, 3)) for input_ in inputs)
        stats_ref = (torch.std_mean(input_[perm], dim=(1, 2, 3)) for input_ in inputs)

        ps = (
            (1.0 / (1.0 + (std0.view(-1, 1) / std1.view(-1, 1)) * (1.0 - v) / v)).view(
                bs, 1, 1, 1
            )
            for (std0, _), (std1, _) in zip(stats, stats_ref)
        )

        labels = kwargs.get("labels")

        if self.soft_label and labels is not None:
            labels = v * labels.float() + (1 - v) * labels[perm].float()
            kwargs["labels"] = labels

        return (
            tuple(
                (
                    (p * input_ + (1.0 - p) * input_ref)
                    / torch.sqrt(torch.pow(p, 2) + torch.pow(1 - p, 2))
                )
                .clamp(0.0, 1.0)
                .where(
                    (probs < self.p)
                    .view(-1, 1, 1, 1)
                    .expand(bs, 3, input_.size(-2), input_.size(-1)),
                    input_,
                )
                for input_, input_ref, p in zip(inputs_org, inputs_ref, ps)
            ),
            kwargs,
        )
