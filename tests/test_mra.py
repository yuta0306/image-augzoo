import os

import torch
from image_augzoo import MRA

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "mra")
os.makedirs(save_to, exist_ok=True)


def test_mra_single():
    mra = MRA()
    image = load_image("assets/image01.jpg", (224, 224)) / 255.0
    processed, kwargs = mra(image)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "mra_single.png"),
        2,
        1.0,
        image,
        processed[0],
    )


def test_mra_single_sr():
    mra = MRA()
    HR = load_image("assets/image01.jpg", (224, 224)) / 255.0
    LR = load_image("assets/image01.jpg", (224, 224), SR=True) / 255.0
    processed, _ = mra(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "mra_single_sr.png"),
        4,
        1.0,
        LR,
        processed[0],
        HR,
        processed[1],
    )


def test_mra_batch():
    mra = MRA()
    images = [
        load_image(f"assets/image0{i}.jpg", (224, 224)) / 255.0 for i in range(1, 7)
    ]
    images = torch.stack(images)

    processed, kwargs = mra(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "mra_batch.png"), 2, 1.0, *tiles)


def test_mra_batch_strategy_low():
    mra = MRA(strategy="low")
    images = [
        load_image(f"assets/image0{i}.jpg", (224, 224)) / 255.0 for i in range(1, 7)
    ]
    images = torch.stack(images)

    processed, kwargs = mra(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "mra_batch_strategy_low.png"), 2, 1.0, *tiles)


def test_mra_batch_mask_ratio08():
    mra = MRA(mask_ratio=0.8)
    images = [
        load_image(f"assets/image0{i}.jpg", (224, 224)) / 255.0 for i in range(1, 7)
    ]
    images = torch.stack(images)

    processed, kwargs = mra(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "mra_batch_mask_ratio08.png"), 2, 1.0, *tiles)


def test_mra_batch_p():
    mra = MRA(p=0.2)
    images = [
        load_image(f"assets/image0{i}.jpg", (224, 224)) / 255.0 for i in range(1, 7)
    ]
    images = torch.stack(images)

    processed, _ = mra(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "mra_batch_p.png"), 2, 1.0, *tiles)
