from typing import Any

import torch


class Transform:
    def __init__(self, *args: Any, p: float = 0.5, **kwargs) -> None:
        ...

    def __call__(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        name = self.__class__.__name__
        params = vars(self)
        params = {k: v for k, v in params.items() if not k.startswith("__")}
        string = f"{name}("
        if len(params) > 0:
            string += "\n"
        for key, value in params.items():
            string += f"    {key}={value}\n"
        string += ")"
        return string
