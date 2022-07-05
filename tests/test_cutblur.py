import os

import torch
from image_augzoo import CutBlur

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "cutblur")
os.makedirs(save_to, exist_ok=True)


def test_cutblur_single():
    cutblur = CutBlur()
    HR = load_image("assets/image01.jpg", (128, 128)) / 255.0
    LR = load_image("assets/image01.jpg", (128, 128), SR=True) / 255.0
    processed, _ = cutblur(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert HR.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "cutblur_single.png"), 3, 1.0, LR, HR, processed[0]
    )


def test_cutblur_batch():
    cutblur = CutBlur()
    HR = [load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)]
    HR = torch.stack(HR)
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    LR = torch.stack(LR)

    processed, _ = cutblur(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert HR.size() == processed[0].size()

    tiles = []
    for i in range(len(HR)):
        tiles.extend([LR[i], HR[i], processed[0][i]])
    save_image(os.path.join(save_to, "cutblur_batch.png"), 3, 1.0, *tiles)


def test_cutblur_batch_p():
    cutblur = CutBlur(p=0.2)
    HR = [load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)]
    HR = torch.stack(HR)
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    LR = torch.stack(LR)

    processed, _ = cutblur(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert HR.size() == processed[0].size()

    tiles = []
    for i in range(len(HR)):
        tiles.extend([LR[i], HR[i], processed[0][i]])
    save_image(os.path.join(save_to, "cutblur_batch_p.png"), 3, 1.0, *tiles)
