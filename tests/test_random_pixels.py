import os

import torch
from image_augzoo import RandomPixels

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "random_pixels")
os.makedirs(save_to, exist_ok=True)


def test_random_pixels_single():
    random_pixels = RandomPixels()
    image = load_image("assets/image01.jpg", (128, 128)) / 255.0
    imgage_ref = load_image("assets/image02.jpg", (128, 128)) / 255.0
    labels = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
    processed, kwargs = random_pixels(image, imgage_ref, labels=labels)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "random_pixels_single.png"),
        2,
        1.0,
        image,
        processed[0],
    )


def test_random_pixels_single_sr():
    random_pixels = RandomPixels()
    HR = load_image("assets/image01.jpg", (128, 128)) / 255.0
    HR_ref = load_image("assets/image04.jpg", (128, 128)) / 255.0
    LR = load_image("assets/image01.jpg", (128, 128), SR=True) / 255.0
    LR_ref = load_image("assets/image04.jpg", (128, 128), SR=True) / 255.0

    processed, _ = random_pixels(LR, LR_ref, HR, HR_ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "random_pixels_single_sr.png"),
        4,
        1.0,
        LR,
        HR,
        processed[0],
        processed[1],
    )


def test_random_pixels_batch():
    random_pixels = RandomPixels()
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)
    labels = torch.nn.functional.one_hot(
        torch.tensor([i for i in range(3)], dtype=torch.long)
    )

    processed, kwargs = random_pixels(images, labels=labels)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "random_pixels_batch.png"), 2, 1.0, *tiles)


def test_random_pixels_batch_p():
    random_pixels = RandomPixels(p=0.2)
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)
    labels = torch.nn.functional.one_hot(
        torch.tensor([i for i in range(3)], dtype=torch.long)
    )

    processed, kwargs = random_pixels(images, labels=labels)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "random_pixels_batch_p.png"), 2, 1.0, *tiles)
