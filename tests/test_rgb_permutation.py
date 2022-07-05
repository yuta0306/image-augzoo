import os

import torch
from image_augzoo import RGBPermutation

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "rgb_permutation")
os.makedirs(save_to, exist_ok=True)


def test_rgb_permutation_single():
    rgb_permutation = RGBPermutation()
    image = load_image("assets/image01.jpg", (128, 128)) / 255.0
    processed, _ = rgb_permutation(image)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "rgb_permutation_single.png"),
        2,
        1.0,
        image,
        processed[0],
    )


def test_rgb_permutation_single_sr():
    rgb_permutation = RGBPermutation()
    HR = load_image("assets/image01.jpg", (128, 128)) / 255.0
    LR = load_image("assets/image01.jpg", (128, 128), SR=True) / 255.0
    processed, _ = rgb_permutation(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "rgb_permutation_single_sr.png"),
        4,
        1.0,
        LR,
        HR,
        processed[0],
        processed[1],
    )


def test_rgb_permutation_batch():
    rgb_permutation = RGBPermutation()
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    images = torch.stack(images)

    processed, _ = rgb_permutation(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    print(processed[0].size())
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "rgb_permutation_batch.png"), 2, 1.0, *tiles)


def test_rgb_permutation_batch_v2():
    rgb_permutation = RGBPermutation()
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    HR = [load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)]
    LR = torch.stack(LR)
    HR = torch.stack(HR)

    processed, _ = rgb_permutation(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()

    tiles = []
    for i in range(len(LR)):
        tiles.extend([LR[i], HR[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "rgb_permutation_batch.png"), 4, 1.0, *tiles)


def test_rgb_permutation_batch_p():
    rgb_permutation = RGBPermutation(p=0.2)
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    HR = [load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)]
    LR = torch.stack(LR)
    HR = torch.stack(HR)

    processed, _ = rgb_permutation(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()

    tiles = []
    for i in range(len(LR)):
        tiles.extend([LR[i], HR[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "rgb_permutation_batch_p.png"), 4, 1.0, *tiles)
