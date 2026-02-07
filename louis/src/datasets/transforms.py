from __future__ import annotations

from typing import Tuple

from torchvision import transforms


def build_transforms(cfg) -> Tuple[transforms.Compose, transforms.Compose]:
    image_size = cfg.get("data", {}).get("image_size", 224)
    resize_size = cfg.get("data", {}).get("resize", 256)

    train_tfms = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_tfms, eval_tfms

