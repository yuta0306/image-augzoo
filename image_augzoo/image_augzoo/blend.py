import torch

from image_augzoo.core.transform import MultiTransform


class Blend(MultiTransform):
    """
    Blend

    Attributes
    ----------
    p : float
    alpha : float
    rgb_range : float
    """

    def __init__(
        self,
        p: float = 1.0,
        alpha: float = 0.6,
        rgb_range: float = 1.0,
    ):
        """
        Parameters
        ----------
        p : float
        alpha : float
        rgb_range : float

        Raises
        ------
        ValueError
            rgb_range is not 1.0 and 255.0
        """
        if rgb_range not in (1.0, 255.0):
            raise ValueError(
                f"rgb_range must be 1.0 or 255.0, but {rgb_range} were given"
            )
        self.alpha = alpha
        self.rgb_range = rgb_range
        super().__init__(p=p)

    def apply(self, *inputs: torch.Tensor, **kwargs):
        if self.alpha <= 0 or torch.rand(1) > self.p:
            return inputs
        device = inputs[0].device
        c = torch.empty((3, 1, 1), device=device).uniform_(0, self.rgb_range)
        refs = (c.repeat((1, input_.size(-2), input_.size(-1))) for input_ in inputs)

        v = torch.empty(1, device=device).uniform_(self.alpha, 1)
        transformed = tuple(
            v * input_ + (1 - v) * ref for (input_, ref) in zip(inputs, refs)
        )

        return transformed

    def apply_batch(self, *inputs: torch.Tensor, **kwargs):
        bs = inputs[0].size(0)
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if self.alpha <= 0 or (probs > self.p).all():
            return inputs
        c = torch.empty((bs, 3, 1, 1), device=device).uniform_(0, self.rgb_range)
        refs = (c.repeat((1, 1, input_.size(-2), input_.size(-1))) for input_ in inputs)

        v = torch.empty(bs, 1, 1, 1, device=device).uniform_(self.alpha, 1)
        transformed = tuple(
            (
                v.repeat(1, input_.size(1), input_.size(2), input_.size(3)) * input_
                + (1 - v).repeat(1, input_.size(1), input_.size(2), input_.size(3))
                * ref
            ).where(
                (probs > self.p)
                .view(-1, 1, 1, 1)
                .expand(bs, input_.size(1), input_.size(2), input_.size(3)),
                input_,
            )
            for input_, ref in zip(inputs, refs)
        )

        return transformed
