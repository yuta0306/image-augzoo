import os

import torch
from image_augzoo import Blend

from tests import BASE_DIR, load_image, save_image

save_to = os.path.join(BASE_DIR, "blend")
os.makedirs(save_to, exist_ok=True)


def test_blend_single():
    blend = Blend()
    image = load_image("assets/image01.jpg", (128, 128)) / 255.0
    processed, _ = blend(image)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(os.path.join(save_to, "blend_single.png"), 2, 1.0, image, processed[0])


def test_blend_batch():
    blend = Blend()
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)

    processed, _ = blend(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "blend_batch.png"), 4, 1.0, *tiles)


def test_blend_batch_v2():
    blend = Blend()
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images_ref = [
        load_image(f"assets/image0{i}.jpg", (256, 256)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)
    images_ref = torch.stack(images_ref)

    processed, _ = blend(images, images_ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert isinstance(processed[1], torch.Tensor)

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], images_ref[i], processed[0][i], processed[1][i]])
    save_image(os.path.join(save_to, "blend_batch_v2.png"), 4, 1.0, *tiles)


def test_blend_batch_p():
    blend = Blend(p=0.2)
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)

    processed, _ = blend(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "blend_batch_p.png"), 4, 1.0, *tiles)


def test_blend_batch_uint():
    blend = Blend(rgb_range=255.0)
    images = [
        load_image(f"assets/image0{i}.jpg", (128, 128)) / 1.0 for i in range(1, 4)
    ]
    images = torch.stack(images)

    processed, _ = blend(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()

    tiles = []
    for i in range(len(images)):
        tiles.extend([images[i], processed[0][i]])
    save_image(os.path.join(save_to, "blend_batch_uint.png"), 4, 255.0, *tiles)
