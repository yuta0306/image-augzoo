import os

import torch
from image_augzoo import CutBlur

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "cutblur")
os.makedirs(save_to, exist_ok=True)


def test_cutblur_single():
    cutblur = CutBlur()
    image = load_image("assets/image01.jpg", (128, 128)) / 255.0
    ref = load_image("assets/image02.jpg", (128, 128)) / 255.0
    processed = cutblur(image, ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "cutblur_single.png"), 3, 1.0, image, ref, processed[0]
    )


def test_cutblur_single_sr():
    cutblur = CutBlur()
    image = load_image("assets/image01.jpg", (128, 128)) / 255.0
    ref = load_image("assets/image02.jpg", (128, 128)) / 255.0
    LR, LR_ref, HR, HR_ref = image, ref, image, ref
    processed = cutblur(LR, LR_ref, HR, HR_ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "cutblur_single_sr.png"),
        3,
        1.0,
        LR,
        LR_ref,
        processed[0],
        HR,
        HR_ref,
        processed[1],
    )


def test_cutblur_batch():
    cutblur = CutBlur()
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)

    processed = cutblur(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "cutblur_batch.png"), 2, 1.0, *tiles)


def test_cutblur_batch_v2():
    cutblur = CutBlur()
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images_ref = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)
    images_ref = torch.stack(images_ref)

    processed = cutblur(images, images_ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert isinstance(processed[1], torch.Tensor)

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], images_ref[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "cutblur_batch_v2.png"), 4, 1.0, *tiles)


def test_cutblur_batch_p():
    cutblur = CutBlur(p=0.2)
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)

    processed = cutblur(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "cutblur_batch_p.png"), 2, 1.0, *tiles)
