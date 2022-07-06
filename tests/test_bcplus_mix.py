import os

import torch
from image_augzoo.image_augzoo.bcplus_mix import BCPlusMix

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "bcplus_mix")
os.makedirs(save_to, exist_ok=True)


def test_bcplus_mix_single():
    bcplus_mix = BCPlusMix()
    image = load_image("assets/image01.jpg", (128, 128)) / 255.0
    ref = load_image("assets/image02.jpg", (128, 128)) / 255.0
    labels = torch.tensor([1, 0], dtype=torch.long)
    labels_ref = torch.tensor([0, 1], dtype=torch.long)
    processed, labels = bcplus_mix(image, ref, labels=torch.stack([labels, labels_ref]))

    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    assert labels["labels"].size(0) == 2
    save_image(
        os.path.join(save_to, "bcplus_mix_single.png"), 3, 1.0, image, ref, processed[0]
    )


def test_bcplus_mix_single_sr():
    bcplus_mix = BCPlusMix()
    HR = load_image("assets/image01.jpg", (128, 128)) / 255.0
    LR = load_image("assets/image01.jpg", (128, 128), SR=True) / 255.0
    HR_ref = load_image("assets/image02.jpg", (128, 128)) / 255.0
    LR_ref = load_image("assets/image02.jpg", (128, 128), SR=True) / 255.0
    processed, _ = bcplus_mix(LR, LR_ref, HR, HR_ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "bcplus_mix_single_sr.png"),
        3,
        1.0,
        LR,
        LR_ref,
        processed[0],
        HR,
        HR_ref,
        processed[1],
    )


def test_bcplus_mix_batch():
    bcplus_mix = BCPlusMix()
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)
    labels = torch.nn.functional.one_hot(
        torch.tensor([0, 1, 2], dtype=torch.long), num_classes=3
    )

    processed, kwargs = bcplus_mix(images, labels=labels)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "bcplus_mix_batch.png"), 2, 1.0, *tiles)


def test_bcplus_mix_batch_v2():
    bcplus_mix = BCPlusMix()
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    HR = [load_image(f"assets/image0{i}.jpg", (256, 256)) / 255.0 for i in range(1, 4)]
    LR = torch.stack(LR)
    HR = torch.stack(HR)

    processed, _ = bcplus_mix(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()

    tiles = []
    for i in range(len(LR)):
        tiles.extend([LR[i], HR[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "bcplus_mix_batch_v2.png"), 4, 1.0, *tiles)


def test_bcplus_mix_batch_p():
    bcplus_mix = BCPlusMix(p=0.2)
    LR = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 4)
    ]
    HR = [load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)]
    LR = torch.stack(LR)
    HR = torch.stack(HR)

    processed, _ = bcplus_mix(LR, HR)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert LR.size() == processed[0].size()

    tiles = []
    for i in range(len(LR)):
        tiles.extend([LR[i], HR[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "bcplus_mix_batch_p.png"), 4, 1.0, *tiles)
