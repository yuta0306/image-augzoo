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
    processed = cutmix(image, ref)
    assert isinstance(processed, tuple)
    assert len(processed) == 1
    assert isinstance(processed[0], torch.Tensor)
    assert image.size() == processed[0].size()
    save_image(
        os.path.join(save_to, "cutmix_single.png"), 3, 1.0, image, ref, processed[0]
    )


# def test_cutmix_single_sr():
#     cutmix = cutmix()
#     image = load_image("assets/image01.jpg", (128, 128)) / 255.0
#     ref = load_image("assets/image02.jpg", (128, 128)) / 255.0
#     LR, LR_ref, HR, HR_ref = image, ref, image, ref
#     processed = cutmix(LR, LR_ref, HR, HR_ref)
#     assert isinstance(processed, tuple)
#     assert len(processed) == 2
#     assert isinstance(processed[0], torch.Tensor)
#     assert image.size() == processed[0].size()
#     save_image(
#         os.path.join(save_to, "cutmix_single_sr.png"),
#         3,
#         1.0,
#         LR,
#         LR_ref,
#         processed[0],
#         HR,
#         HR_ref,
#         processed[1],
#     )


# def test_cutmix_batch():
#     cutmix = cutmix()
#     images = [
#         load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
#     ]
#     images = torch.stack(images)

#     processed = cutmix(images)
#     assert isinstance(processed, tuple)
#     assert len(processed) == 1
#     assert isinstance(processed[0], torch.Tensor)
#     assert images.size() == processed[0].size()

#     tiles = []
#     for i in range(len(images)):
#         tiles.extend([images[i], processed[0][i]])
#     save_image(os.path.join(save_to, "cutmix_batch.png"), 2, 1.0, *tiles)


# def test_cutmix_batch_v2():
#     cutmix = cutmix()
#     images = [
#         load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
#     ]
#     images_ref = [
#         load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
#     ]
#     images = torch.stack(images)
#     images_ref = torch.stack(images_ref)

#     processed = cutmix(images, images_ref)
#     assert isinstance(processed, tuple)
#     assert len(processed) == 2
#     assert isinstance(processed[0], torch.Tensor)
#     assert isinstance(processed[1], torch.Tensor)

#     tiles = []
#     for i in range(len(images)):
#         tiles.extend([images[i], images_ref[i], processed[0][i], processed[1][i]])
#     save_image(os.path.join(save_to, "cutmix_batch_v2.png"), 4, 1.0, *tiles)


# def test_cutmix_batch_p():
#     cutmix = cutmix(p=0.2)
#     images = [
#         load_image(f"assets/image0{i}.jpg", (128, 128)) / 255.0 for i in range(1, 4)
#     ]
#     images = torch.stack(images)

#     processed = cutmix(images)
#     assert isinstance(processed, tuple)
#     assert len(processed) == 1
#     assert isinstance(processed[0], torch.Tensor)
#     assert images.size() == processed[0].size()

#     tiles = []
#     for i in range(len(images)):
#         tiles.extend([images[i], processed[0][i]])
#     save_image(os.path.join(save_to, "cutmix_batch_p.png"), 2, 1.0, *tiles)
