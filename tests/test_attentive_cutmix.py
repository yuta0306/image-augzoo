import os

import torch
from image_augzoo import AttentiveCutMix

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "attentive_cutmix")
os.makedirs(save_to, exist_ok=True)


def test_attentive_cutmix_single():
    attentive_cutmix = AttentiveCutMix()
    image = load_image("assets/image01.jpg", (224, 224)) / 255.0
    ref = load_image("assets/image06.jpg", (224, 224)) / 255.0
    processed, _ = attentive_cutmix(image, ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "attentive_cutmix_single.png"),
        3,
        1.0,
        image,
        ref,
        processed[0],
    )


def test_attentive_cutmix_single_sr():
    attentive_cutmix = AttentiveCutMix()
    HR = load_image("assets/image01.jpg", (224, 224)) / 255.0
    LR = load_image("assets/image01.jpg", (224, 224), SR=True) / 255.0
    HR_ref = load_image("assets/image06.jpg", (224, 224)) / 255.0
    LR_ref = load_image("assets/image06.jpg", (224, 224), SR=True) / 255.0
    processed, _ = attentive_cutmix(LR, LR_ref, HR, HR_ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "attentive_cutmix_single_sr.png"),
        3,
        1.0,
        LR,
        LR_ref,
        processed[0],
        HR,
        HR_ref,
        processed[1],
    )


def test_attentive_cutmix_batch():
    attentive_cutmix = AttentiveCutMix()
    images = [
        load_image(f"assets/image0{i}.jpg", (224, 224), SR=True) / 255.0
        for i in range(1, 7)
    ]
    images = torch.stack(images)

    processed, _ = attentive_cutmix(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "attentive_cutmix_batch.png"), 4, 1.0, *tiles)


def test_attentive_cutmix_batch_p():
    attentive_cutmix = AttentiveCutMix(p=0.2)
    images = [
        load_image(f"assets/image0{i}.jpg", (224, 224), SR=True) / 255.0
        for i in range(1, 7)
    ]
    images = torch.stack(images)

    processed, _ = attentive_cutmix(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "attentive_cutmix_batch_p.png"), 4, 1.0, *tiles)
