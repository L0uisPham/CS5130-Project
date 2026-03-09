from __future__ import annotations


from src.models.base import BaseModelAdapter
from src.models.external.convnext_timm import ConvNeXtTimmAdapter
from src.models.huggingface.hf_image import HfImageClassifierAdapter
from src.models.torchvision.convnext import ConvNeXtTinyAdapter
from src.models.torchvision.vgg19 import VGG19Adapter


def build_model(cfg) -> BaseModelAdapter:
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name")
    pretrained = bool(model_cfg.get("pretrained", True))
    num_classes = int(cfg.get("num_classes", len(cfg.get("labels", []))))

    if name == "convnext_t":
        return ConvNeXtTinyAdapter(num_classes=num_classes, pretrained=pretrained)
    if name in {"vgg19", "vgg19_hf", "hf_vgg19"}:
        return VGG19Adapter(num_classes=num_classes, pretrained=pretrained)
    if name == "convnext_timm":
        timm_name = model_cfg.get("timm_name", "convnext_tiny")
        return ConvNeXtTimmAdapter(
            model_name=timm_name, num_classes=num_classes, pretrained=pretrained
        )
    if name in {"hf_resnet34", "resnet34"}:
        hf_name = model_cfg.get("hf_name", "microsoft/resnet-34")
        return HfImageClassifierAdapter(
            model_name=hf_name, num_classes=num_classes, pretrained=pretrained
        )
    if name == "hf_vgg19_unsupported":
        hf_name = model_cfg.get("hf_name", "keras/vgg_19_imagenet")
        return HfImageClassifierAdapter(
            model_name=hf_name, num_classes=num_classes, pretrained=pretrained
        )
    if name in {"hf_efficientnet_b4", "efficientnet_b4"}:
        hf_name = model_cfg.get("hf_name", "google/efficientnet-b4")
        return HfImageClassifierAdapter(
            model_name=hf_name, num_classes=num_classes, pretrained=pretrained
        )
    if name in {"hf_deit_small", "deit_small"}:
        hf_name = model_cfg.get("hf_name", "facebook/deit-small-patch16-224")
        return HfImageClassifierAdapter(
            model_name=hf_name, num_classes=num_classes, pretrained=pretrained
        )
    if name in {"hf_swin_tiny", "swin_tiny"}:
        hf_name = model_cfg.get("hf_name", "microsoft/swin-tiny-patch4-window7-224")
        return HfImageClassifierAdapter(
            model_name=hf_name, num_classes=num_classes, pretrained=pretrained
        )

    raise ValueError(f"Unknown model adapter: {name}")
