import os

import torch
from image_augzoo import Cutout

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "cutout")
os.makedirs(save_to, exist_ok=True)


def test_cutout_single():
    cutout = Cutout()
    image = load_image("assets/image01.jpg", (128, 128)) / 255.0
    processed = cutout(image)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "cutout_single.png"),
        2,
        1.0,
        image,
        processed[0],
    )


def test_cutout_single_sr():
    cutout = Cutout()
    HR = load_image("assets/image01.jpg", (128, 128)) / 255.0
    LR = load_image("assets/image01.jpg", (128, 128), SR=True) / 255.0
    processed = cutout(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "cutout_single_sr.png"),
        4,
        1.0,
        LR,
        HR,
        processed[0],
        processed[1],
    )


def test_cutout_batch():
    cutout = Cutout()
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    HR = [load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)]
    LR = torch.stack(LR)
    HR = torch.stack(HR)

    processed = cutout(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()

    tiles = []
    for i in range(len(LR)):
        tiles.extend([LR[i], HR[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "cutout_batch.png"), 4, 1.0, *tiles)


def test_cutout_batch_p():
    cutout = Cutout(p=0.2)
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    HR = [load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)]
    LR = torch.stack(LR)
    HR = torch.stack(HR)

    processed = cutout(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()

    tiles = []
    for i in range(len(LR)):
        tiles.extend([LR[i], HR[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "cutout_batch_p.png"), 4, 1.0, *tiles)
