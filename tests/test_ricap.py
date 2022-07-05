import os

import torch
from image_augzoo.image_augzoo.ricap import RICAP

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "ricap")
os.makedirs(save_to, exist_ok=True)


def test_ricap_single():
    ricap = RICAP()
    image = load_image("assets/image01.jpg", (128, 128)) / 255.0
    image2 = load_image("assets/image02.jpg", (128, 128)) / 255.0
    image3 = load_image("assets/image03.jpg", (128, 128)) / 255.0
    image4 = load_image("assets/image04.jpg", (128, 128)) / 255.0
    processed, _ = ricap(image, image2, image3, image4)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "ricap_single.png"),
        5,
        1.0,
        image,
        image2,
        image3,
        image4,
        processed[0],
    )


def test_ricap_batch():
    ricap = RICAP()
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 7)
    ]
    images = torch.stack(images)

    processed, _ = ricap(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "ricap_batch.png"), 4, 1.0, *tiles)


def test_ricap_batch_v2():
    ricap = RICAP()
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 5)
    ]
    images_ref = [
        load_image(f"assets/image0{i}.jpg", (128, 128), SR=True) / 255.0
        for i in range(1, 5)
    ]
    images = torch.stack(images)
    images_ref = torch.stack(images_ref)

    processed, _ = ricap(images, images_ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert isinstance(processed[1], torch.Tensor)

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], images_ref[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "ricap_batch_v2.png"), 4, 1.0, *tiles)


def test_ricap_batch_p():
    ricap = RICAP(p=0.2)
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)

    processed, _ = ricap(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "ricap_batch_p.png"), 4, 1.0, *tiles)
