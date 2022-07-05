from typing import Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from image_augzoo.core.transform import MultiTransform


class AttentiveCutMix(MultiTransform):
    """
    Attentive CutMix
    Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning Based Image Classification
    [Arxiv(https://arxiv.org/pdf/2003.13048.pdf)](https://arxiv.org/pdf/2003.13048.pdf)

    Attributes
    ----------
    model : Any
    patch_size : tuple[int, int], default=(7, 7)
    grid_size : tuple[int, int], default=(32, 32)
    top_k : int, default=6
    p : float, default=1.0
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (7, 7),
        grid_size: Tuple[int, int] = (32, 32),
        top_k: int = 6,
        p: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str, default='resnet50'
        patch_size : tuple[int, int], default=(7, 7)
        grid_size : tuple[int, int], default=(32, 32)
        top_k : int, default=6
        p : float, default=1.0
        """
        model = torchvision.models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.image_size = tuple(
            patch * grid for patch, grid in zip(patch_size, grid_size)
        )
        self.top_k = top_k
        super().__init__(p=p)

    @torch.inference_mode()
    def _get_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        last_feature_map = self.model(inputs)[:, -1, :, :]
        return last_feature_map

    def _get_top_k_region(self, inputs: torch.Tensor) -> torch.Tensor:
        bs = inputs.size(0)
        values, indices = inputs.view(bs, -1).topk(k=self.top_k, dim=-1)
        region = torch.zeros((bs, 3, *self.image_size))
        for i, indice in enumerate(indices):
            for idx in indice:
                region[i][
                    ...,
                    idx * self.grid_size[0] : (idx + 1) * self.grid_size[0],
                    idx * self.grid_size[1] : (idx + 1) * self.grid_size[1],
                ] = 1
        return region

    def apply(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        if torch.rand(1) > self.p:
            return inputs, kwargs

        sizes = (input_.size()[-2:] for input_ in inputs[::2])
        inputs_org = (
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)(input_)
            for input_ in inputs[::2]
        )
        inputs_ref = (
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)(input_)
            for input_ in inputs[1::2]
        )

        features_map = self._get_feature_map(inputs[0].unsqueeze(dim=0))
        region = self._get_top_k_region(features_map)[0]

        return (
            tuple(
                T.Resize(size=size, interpolation=T.InterpolationMode.NEAREST)(
                    region * input_org + (1 - region) * input_ref
                )
                for input_org, input_ref, size in zip(inputs_org, inputs_ref, sizes)
            ),
            kwargs,
        )

    def apply_batch(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        bs = inputs[0].size(0)
        device = inputs[0].device
        probs = torch.rand(bs, device=device)
        if (probs > self.p).all():
            return inputs, kwargs

        sizes = (input_.size()[-2:] for input_ in inputs)
        perm = torch.randperm(bs)
        inputs_org = (
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)(input_)
            for input_ in inputs
        )
        inputs_ref = (
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)(
                input_[perm]
            )
            for input_ in inputs
        )

        features_map = self._get_feature_map(
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)(
                inputs[0][perm]
            )
        )
        region = self._get_top_k_region(features_map)

        return (
            tuple(
                T.Resize(size=size, interpolation=T.InterpolationMode.NEAREST)(
                    region * input_ + (1 - region) * input_ref
                ).where(
                    (probs < self.p)
                    .view(-1, 1, 1, 1)
                    .expand(bs, 3, input_.size(-2), input_.size(-1)),
                    input_,
                )
                for input_, input_ref, size in zip(inputs_org, inputs_ref, sizes)
            ),
            kwargs,
        )
