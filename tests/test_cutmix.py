import os

import torch
from image_augzoo import CutMix

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "cutmix")
os.makedirs(save_to, exist_ok=True)


def test_cutmix_single():
    cutmix = CutMix()
    image = load_image("assets/image01.jpg", (128, 128)) / 255.0
    ref = load_image("assets/image02.jpg", (128, 128)) / 255.0
    processed, _ = cutmix(image, ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "cutmix_single.png"), 3, 1.0, image, ref, processed[0]
    )


def test_cutmix_single_sr():
    cutmix = CutMix()
    HR = load_image("assets/image01.jpg", (128, 128)) / 255.0
    LR = load_image("assets/image01.jpg", (128, 128), SR=True) / 255.0
    HR_ref = load_image("assets/image02.jpg", (128, 128)) / 255.0
    LR_ref = load_image("assets/image02.jpg", (128, 128), SR=True) / 255.0
    processed, _ = cutmix(LR, LR_ref, HR, HR_ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "cutmix_single_sr.png"),
        3,
        1.0,
        LR,
        LR_ref,
        processed[0],
        HR,
        HR_ref,
        processed[1],
    )


def test_cutmix_batch():
    cutmix = CutMix()
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    HR = [load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)]
    LR = torch.stack(LR)
    HR = torch.stack(HR)

    processed, _ = cutmix(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()

    tiles = []
    for i in range(len(LR)):
        tiles.extend([LR[i], HR[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "cutmix_batch.png"), 4, 1.0, *tiles)


def test_cutmix_batch_v2():
    cutmix = CutMix()
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    HR = [load_image(f"assets/image0{i}.jpg", (256, 256)) / 255.0 for i in range(1, 4)]
    LR = torch.stack(LR)
    HR = torch.stack(HR)

    processed, _ = cutmix(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()

    tiles = []
    for i in range(len(LR)):
        tiles.extend([LR[i], HR[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "cutmix_batch_v2.png"), 4, 1.0, *tiles)


def test_cutmix_batch_p():
    cutmix = CutMix(p=0.2)
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    HR = [load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)]
    LR = torch.stack(LR)
    HR = torch.stack(HR)

    processed, _ = cutmix(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()

    tiles = []
    for i in range(len(LR)):
        tiles.extend([LR[i], HR[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "cutmix_batch_p.png"), 4, 1.0, *tiles)
