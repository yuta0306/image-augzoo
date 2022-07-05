from typing import List, Tuple

import torch
from image_augzoo.core.transform import Transform


class Compose:
    transforms: List[Transform]

    def __init__(self, transforms: List[Transform]) -> None:
        assert len(transforms) > 0
        self.transforms = transforms

    def __call__(
        self, *inputs: torch.Tensor, **kwargs
    ) -> Tuple[Tuple[torch.Tensor, ...], dict]:
        for transform in self.transforms:
            inputs, kwargs = transform(*inputs, **kwargs)
        return inputs, kwargs

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, idx: int):
        return self.transforms[idx]

    def __repr__(self) -> str:
        string = "Compose([\n"
        for transform in self.transforms:
            string += f"  {transform}\n"
        string += "])"
        return string
