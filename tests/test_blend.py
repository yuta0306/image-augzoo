import cv2
import torch
from image_augzoo import Blend


def load_image(path: str, size: tuple[int, int]):
    image = cv2.imread(path, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = torch.from_numpy(image)
    image = image.transpose(0, -1)
    return image


def test_blend_single():
    blend = Blend()
    image = load_image("images/image01.jpg", (128, 128)) / 255.0
    processed = blend(image)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()


def test_blend_batch():
    blend = Blend()
    images = [
        load_image(f"images/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)

    processed = blend(images)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert images.size() == processed[0].size()


def test_blend_batch_v2():
    blend = Blend()
    images = [
        load_image(f"images/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
    ]
    images_ref = [
        load_image(f"images/image0{i}.jpg", (256, 256)) / 255.0 for i in range(1, 4)
    ]
    images = torch.stack(images)
    images_ref = torch.stack(images_ref)

    processed = blend(images, images_ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 2
    assert isinstance(processed[0], torch.Tensor)
    assert isinstance(processed[1], torch.Tensor)
