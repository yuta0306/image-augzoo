import torch
from image_augzoo.core.transform import Transform


class Compose:
    transforms: list

    def __init__(self, *transforms: Transform) -> None:
        assert len(transforms) > 0
        self.transforms = list(transforms)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            inputs = transform(inputs)
        return inputs

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
