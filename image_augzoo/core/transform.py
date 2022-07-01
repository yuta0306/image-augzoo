import warnings
from typing import Any

import torch


class Transform:
    def __init__(self, *args: Any, p: float = 1.0, **kwargs) -> None:
        assert 0.0 <= p <= 1.0
        self.p = p

    def apply(self, *inputs: torch.Tensor, **kwargs):
        raise NotImplementedError

    def apply_batch(self, *inputs: torch.Tensor, **kwargs):
        raise NotImplementedError

    def __call__(self, *inputs: torch.Tensor, **kwargs):
        is_batched = inputs[0].ndim == 4
        if is_batched:
            try:
                return self.apply_batch(*inputs, **kwargs)
            except NotImplementedError:
                warnings.warn(
                    "`apply` is called because `apply_batch` is not implemented",
                    UserWarning,
                )
                return self.apply(*inputs, **kwargs)
        return self.apply(*inputs, **kwargs)

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


class DualTransform(Transform):
    def __init__(self, *args: Any, p: float = 1.0, **kwargs) -> None:
        super().__init__(*args, p=p, **kwargs)

    def __call__(self, *inputs: torch.Tensor, **kwargs):
        assert len(inputs) == 2
        return super().__call__(*inputs, **kwargs)

    def __repr__(self) -> str:
        return super().__repr__()


class MultiTransform(Transform):
    def __init__(self, *args: Any, p: float = 1.0, **kwargs) -> None:
        super().__init__(*args, p=p, **kwargs)

    def __call__(self, *inputs: torch.Tensor, **kwargs):
        return super().__call__(*inputs, **kwargs)

    def __repr__(self) -> str:
        return super().__repr__()
