from typing import Tuple

import timm
import torch
import torchvision.transforms as T

from image_augzoo.core.transform import MultiTransform


class MRA(MultiTransform):
    """
    MRA
    Masked Autoencoders are Robust Data Augmentors
    [Arxiv](https://arxiv.org/pdf/2206.04846v1.pdf)

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
        model_name: str = "vit_base_patch16_224",
        patch_size: Tuple[int, int] = (14, 14),
        grid_size: Tuple[int, int] = (16, 16),
        mask_ratio: float = 0.4,
        strategy: str = "high",  # ["high", "low"]
        p: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str, default='resnet50'
        patch_size : tuple[int, int], default=(14, 14)
        grid_size : tuple[int, int], default=(16, 16)
        mask_ratio : float, default=0.75
        strategy : str, default='high'
        p : float, default=1.0
        """
        assert strategy in ("high", "low")
        self.model = timm.create_model(model_name, pretrained=True)
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.image_size = tuple(
            patch * grid for patch, grid in zip(patch_size, grid_size)
        )
        self.mask_ratio = mask_ratio
        self.strategy = strategy
        self.total = 1
        for num in patch_size:
            self.total *= num
        self.top_k = int(self.total * self.mask_ratio)
        super().__init__(p=p)

    @torch.inference_mode()
    def _get_last_hidden_states(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print(inputs.size())
        x = self.model.patch_embed(inputs)
        if self.model.cls_token is not None:
            x = torch.cat((self.model.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        cls_token = x[:, 0, :].unsqueeze(dim=1)
        last_hidden_states = x[:, 1:, ...]
        return cls_token, last_hidden_states

    def _compute_attention(
        self, cls_token: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        `cls_token` is used as queries
        `hidden_states` is used as keys
        """
        attention_score = torch.bmm(cls_token, hidden_states.transpose(-1, -2))
        return attention_score[:, 0, :]  # (B, 1, 768) -> (B, 768)

    def _get_top_k_region(
        self,
        cls_token: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = cls_token.size(0)
        attention_score = self._compute_attention(
            cls_token=cls_token,
            hidden_states=hidden_states,
        )
        if self.strategy == "low":
            attention_score = -1 * attention_score
        values, indices = attention_score.topk(k=self.top_k, dim=-1)
        region = torch.zeros((bs, 3, *self.image_size))
        i = 0
        k_sizes = []
        for i, indice in enumerate(indices):
            if self.strategy == "high":
                k_sizes.append(len(indice))
            else:
                k_sizes.append(self.total - len(indice))
            for idx in indice:
                left = (idx % self.patch_size[0]) * self.grid_size[0]
                top = (idx // self.patch_size[1]) * self.grid_size[1]
                region[i][
                    ..., left : left + self.grid_size[0], top : top + self.grid_size[1]
                ] = 1
        top_k = torch.tensor(k_sizes, dtype=torch.long, device=cls_token.device)
        return region, top_k

    def apply(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        if torch.rand(1) > self.p:
            return inputs, kwargs

        sizes = (input_.size()[-2:] for input_ in inputs)
        inputs = (
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)(
                inputs[0]
            ),
            *inputs[1:],
        )

        cls_token, last_hidden_states = self._get_last_hidden_states(
            inputs[0].unsqueeze(dim=1)
        )
        mask, replaced = self._get_top_k_region(
            cls_token=cls_token, hidden_states=last_hidden_states
        )
        mask = mask[0]  # 1 means mask
        replaced = replaced[0]

        return (
            tuple(
                T.Resize(size=size, interpolation=T.InterpolationMode.NEAREST)(
                    (1 - mask) * input_
                )
                for input_, size in zip(inputs, sizes)
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
        inputs = (
            T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)(
                inputs[0]
            ),
            *inputs[1:],
        )

        cls_token, last_hidden_states = self._get_last_hidden_states(inputs[0])
        mask, replaced = self._get_top_k_region(
            cls_token=cls_token, hidden_states=last_hidden_states
        )
        replaced = replaced.view(bs, 1)

        return (
            tuple(
                T.Resize(size=size, interpolation=T.InterpolationMode.NEAREST)(
                    (1 - mask) * input_
                ).where(
                    (probs < self.p)
                    .view(-1, 1, 1, 1)
                    .expand(bs, 3, input_.size(-2), input_.size(-1)),
                    input_,
                )
                for input_, size in zip(inputs, sizes)
            ),
            kwargs,
        )
