from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import torch


def load_image(path: str, size: Tuple[int, int], SR: bool = False):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    if SR:
        image = cv2.resize(image, (size[0] // 4, size[1] // 4))
        image = cv2.resize(image, size)
    image = torch.from_numpy(image)
    image = image.transpose(1, 2).transpose(0, 1)
    return image


def save_image(
    path: str, cols: int = 2, range_: float = 1.0, *image: torch.Tensor
) -> None:
    plt.clf()
    fig = make_tile(*image, cols=cols, range_=range_)
    fig.savefig(path)


def make_tile(*image: torch.Tensor, cols: int = 2, range_: float = 1.0):
    images = [
        (img / range_ * 255).transpose(0, 1).transpose(1, 2).to(torch.uint8).numpy()
        for img in image
    ]
    rows = max(1, len(images) // cols)
    fig, axes = plt.subplots(rows, cols)
    for ax, image in zip(axes.flatten(), images):
        ax.imshow(image)
    plt.tight_layout()
    return fig
