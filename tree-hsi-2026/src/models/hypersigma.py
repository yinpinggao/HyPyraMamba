"""Thin adapter for the author-released HyperSIGMA classification model.

The learned network remains ``SSFusionFramework`` from the pinned HyperSIGMA
checkout. This module only supplies import compatibility, the project BCHW
contract, and official classification-style checkpoint loading for train-only
PCA30, 33x33 competition patches.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "HyperSIGMA"
UPSTREAM_MODEL_DIR = UPSTREAM_ROOT / "ImageClassification" / "model"
UPSTREAM_SOURCE = UPSTREAM_MODEL_DIR / "ss_fusion_cls.py"
UPSTREAM_SEG_SOURCE = UPSTREAM_MODEL_DIR / "ss_fusion_seg.py"
UPSTREAM_COMMIT = "07e9ea24e3072fcb5c3a92a2bcb8185e43b295b9"
DEFAULT_SPATIAL_CHECKPOINT = PROJECT_ROOT / "weights" / "hypersigma" / "spat-vit-base-ultra-checkpoint-1599.pth"
DEFAULT_SPECTRAL_CHECKPOINT = PROJECT_ROOT / "weights" / "hypersigma" / "spec-vit-base-ultra-checkpoint-1599.pth"


def _load_source(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream HyperSIGMA source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _upstream_framework_class() -> type[nn.Module]:
    """Import the unchanged author class despite its notebook-style imports."""
    if not UPSTREAM_SOURCE.is_file():
        raise FileNotFoundError(f"Missing upstream HyperSIGMA source: {UPSTREAM_SOURCE}")
    package_name = "_tree_hsi_upstream_hypersigma_model"
    package = ModuleType(package_name)
    package.__path__ = [str(UPSTREAM_MODEL_DIR)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    old_mmengine = sys.modules.get("mmengine")
    old_mmengine_dist = sys.modules.get("mmengine.dist")
    if old_mmengine_dist is None:
        mmengine = ModuleType("mmengine")
        mmengine_dist = ModuleType("mmengine.dist")
        mmengine_dist.get_dist_info = lambda: (0, 1)
        mmengine.dist = mmengine_dist
        sys.modules["mmengine"] = mmengine
        sys.modules["mmengine.dist"] = mmengine_dist

    old_model = sys.modules.get("model")
    sys.modules["model"] = package
    try:
        for stem in ("SpatViT_fusion_patch", "SpecViT_fusion"):
            module = _load_source(f"{package_name}.{stem}", UPSTREAM_MODEL_DIR / f"{stem}.py")
            setattr(package, stem, module)
        framework = _load_source(f"{package_name}.ss_fusion_cls", UPSTREAM_SOURCE)
    finally:
        if old_model is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = old_model
        if old_mmengine_dist is None:
            sys.modules.pop("mmengine.dist", None)
            if old_mmengine is None:
                sys.modules.pop("mmengine", None)
            else:
                sys.modules["mmengine"] = old_mmengine
    return framework.SSFusionFramework


@lru_cache(maxsize=1)
def _upstream_segmentation_framework_class() -> type[nn.Module]:
    """Import the pinned author dense framework without changing its modules."""
    if not UPSTREAM_SEG_SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing upstream HyperSIGMA source: {UPSTREAM_SEG_SOURCE}"
        )
    package_name = "_tree_hsi_upstream_hypersigma_seg_model"
    package = ModuleType(package_name)
    package.__path__ = [str(UPSTREAM_MODEL_DIR)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    old_mmengine = sys.modules.get("mmengine")
    old_mmengine_dist = sys.modules.get("mmengine.dist")
    if old_mmengine_dist is None:
        mmengine = ModuleType("mmengine")
        mmengine_dist = ModuleType("mmengine.dist")
        mmengine_dist.get_dist_info = lambda: (0, 1)
        mmengine.dist = mmengine_dist
        sys.modules["mmengine"] = mmengine
        sys.modules["mmengine.dist"] = mmengine_dist

    old_model = sys.modules.get("model")
    sys.modules["model"] = package
    try:
        for stem in ("SpatViT_fusion", "SpecViT_fusion"):
            module = _load_source(
                f"{package_name}.{stem}", UPSTREAM_MODEL_DIR / f"{stem}.py"
            )
            setattr(package, stem, module)
        framework = _load_source(
            f"{package_name}.ss_fusion_seg", UPSTREAM_SEG_SOURCE
        )
    finally:
        if old_model is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = old_model
        if old_mmengine_dist is None:
            sys.modules.pop("mmengine.dist", None)
            if old_mmengine is None:
                sys.modules.pop("mmengine", None)
            else:
                sys.modules["mmengine"] = old_mmengine
    return framework.SSFusionFramework


def _checkpoint_model(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing official HyperSIGMA checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict) or not state:
        raise TypeError(f"No model state dictionary in {path}")
    if next(iter(state)).startswith("module."):
        state = {key[7:]: value for key, value in state.items()}
    return state


class HyperSIGMA(nn.Module):
    """Project interface around the official two-branch HyperSIGMA ViT-B."""

    def __init__(
        self, in_channels: int = 98, patch_size: int = 33,
        num_classes: int = 17, spatial_patch_size: int = 2,
        spatial_checkpoint: str | Path = DEFAULT_SPATIAL_CHECKPOINT,
        spectral_checkpoint: str | Path = DEFAULT_SPECTRAL_CHECKPOINT,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 30:
            raise ValueError(
                f"M5 phase-1 uses train-only PCA30 input, got {in_channels} bands"
            )
        if patch_size != 33 or spatial_patch_size != 2:
            raise ValueError("The released classification recipe uses 33x33 patches and spatial patch size 2")
        upstream_class = _upstream_framework_class()
        self.upstream = upstream_class(
            img_size=patch_size, in_channels=in_channels, patch_size=spatial_patch_size,
            classes=num_classes, model_size="base",
        )
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.num_classes = int(num_classes)
        self.spatial_patch_size = int(spatial_patch_size)
        self.gradient_checkpointing = True
        self.pretrained_report: dict[str, object] = {}
        if load_pretrained:
            self.pretrained_report = self.load_official_weights(Path(spatial_checkpoint), Path(spectral_checkpoint))

    @property
    def halo(self) -> int:
        return self.patch_size // 2

    def load_official_weights(self, spatial_path: Path, spectral_path: Path) -> dict[str, object]:
        target = self.upstream.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        spatial = _checkpoint_model(spatial_path)
        for key, value in spatial.items():
            target_key = f"spat_encoder.{key}"
            if target_key not in target:
                continue
            if (
                key in {"patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed"}
                or "spat_map" in key
                or "spat_output_maps" in key
            ):
                skipped.append(target_key)
                continue
            if value.shape == target[target_key].shape:
                loaded[target_key] = value
        del spatial

        spectral = _checkpoint_model(spectral_path)
        for key, value in spectral.items():
            target_key = f"spec_encoder.{key}"
            if "patch_embed" in key or "spat_map" in key or key == "fpn1.0.weight":
                skipped.append(target_key)
                continue
            if target_key in target and value.shape == target[target_key].shape:
                loaded[target_key] = value
        del spectral
        incompatible = self.upstream.load_state_dict(loaded, strict=False)
        return {
            "loaded_tensors": len(loaded), "adapted_tensors": (),
            "skipped_tensors": len(skipped),
            "skipped_tensor_names": tuple(skipped),
            "missing_tensors": len(incompatible.missing_keys),
            "unexpected_tensors": tuple(incompatible.unexpected_keys),
            "spatial_checkpoint": str(spatial_path), "spectral_checkpoint": str(spectral_path),
        }

    def set_backbone_trainable(self, trainable: bool) -> None:
        for encoder in (self.upstream.spat_encoder, self.upstream.spec_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = bool(trainable)

    def set_peft_trainable(self) -> None:
        """Train only task adapters plus LayerNorm/bias parameters."""
        for parameter in self.parameters():
            parameter.requires_grad = False
        for name, parameter in self.named_parameters():
            is_head = not (
                name.startswith("upstream.spat_encoder.")
                or name.startswith("upstream.spec_encoder.")
            )
            is_task_embedding = ".patch_embed." in name or ".pos_embed" in name
            is_norm_or_bias = name.endswith(".bias") or ".norm" in name or ".ln" in name
            if is_head or is_task_embedding or is_norm_or_bias:
                parameter.requires_grad = True

    def peft_parameter_groups(
        self, head_lr: float, patch_lr: float, norm_lr: float
    ) -> list[dict[str, object]]:
        groups: dict[str, list[nn.Parameter]] = {
            "head": [],
            "patch_embedding": [],
            "norm_bias": [],
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if not (
                name.startswith("upstream.spat_encoder.")
                or name.startswith("upstream.spec_encoder.")
            ):
                groups["head"].append(parameter)
            elif ".patch_embed." in name or ".pos_embed" in name:
                groups["patch_embedding"].append(parameter)
            else:
                groups["norm_bias"].append(parameter)
        return [
            {"params": groups["head"], "lr": head_lr},
            {"params": groups["patch_embedding"], "lr": patch_lr},
            {"params": groups["norm_bias"], "lr": norm_lr},
        ]

    def backbone_parameters(self):
        yield from self.upstream.spat_encoder.parameters()
        yield from self.upstream.spec_encoder.parameters()

    def head_parameters(self):
        backbone_ids = {id(parameter) for parameter in self.backbone_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in backbone_ids:
                yield parameter

    def forward(
        self, x: torch.Tensor, interpolate_pos_encoding: bool = True
    ) -> torch.Tensor:
        expected = (self.in_channels, self.patch_size, self.patch_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"HyperSIGMA expects [batch, {expected}], got {tuple(x.shape)}")
        if not interpolate_pos_encoding:
            raise ValueError(
                "M5 requires interpolate_pos_encoding=True for both training and evaluation"
            )
        # The official classification recipe initializes the task-specific
        # PCA30 patch projection and 33x33 position tensor for fine-tuning,
        # while loading the compatible pretrained transformer blocks.
        return self.upstream(x)


class HyperSIGMADense(nn.Module):
    """Thin dual-view adapter around the official dense HyperSIGMA framework.

    The author model is kept intact.  Only its published forward is expanded so
    the spatial encoder can consume PCA30 while the spectral encoder sees the
    ordered 98-band spectrum.  Decoder and fusion layers are the original
    ``ss_fusion_seg.SSFusionFramework`` modules.
    """

    def __init__(
        self,
        spatial_channels: int = 30,
        spectral_channels: int = 98,
        image_size: int = 128,
        num_classes: int = 17,
        spatial_patch_size: int = 8,
        spatial_checkpoint: str | Path = DEFAULT_SPATIAL_CHECKPOINT,
        spectral_checkpoint: str | Path = DEFAULT_SPECTRAL_CHECKPOINT,
    ) -> None:
        super().__init__()
        framework = _upstream_segmentation_framework_class()
        self.upstream = framework(
            img_size=image_size,
            in_channels=spatial_channels,
            patch_size=spatial_patch_size,
            classes=num_classes,
            model_size="base",
        )
        self.spatial_channels = int(spatial_channels)
        self.spectral_channels = int(spectral_channels)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.spatial_patch_size = int(spatial_patch_size)
        self.pretrained_report = self.load_encoder_weights(
            Path(spatial_checkpoint), Path(spectral_checkpoint)
        )

    def load_encoder_weights(
        self, spatial_path: Path, spectral_path: Path
    ) -> dict[str, object]:
        target = self.upstream.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        sources = (
            ("spat_encoder", _checkpoint_model(spatial_path)),
            ("spec_encoder", _checkpoint_model(spectral_path)),
        )
        for prefix, state in sources:
            for key, value in state.items():
                target_key = f"{prefix}.{key}"
                if target_key in target and target[target_key].shape == value.shape:
                    loaded[target_key] = value
                elif target_key in target:
                    skipped.append(target_key)
        incompatible = self.upstream.load_state_dict(loaded, strict=False)
        return {
            "loaded_tensors": len(loaded),
            "shape_skipped_tensors": len(skipped),
            "missing_tensors": len(incompatible.missing_keys),
            "unexpected_tensors": tuple(incompatible.unexpected_keys),
            "spatial_checkpoint": str(spatial_path),
            "spectral_checkpoint": str(spectral_path),
        }

    def forward(
        self, spatial_x: torch.Tensor, spectral_x: torch.Tensor
    ) -> torch.Tensor:
        if spatial_x.ndim != 4 or spectral_x.ndim != 4:
            raise ValueError("HyperSIGMADense inputs must both be BCHW")
        if spatial_x.shape[0] != spectral_x.shape[0] or spatial_x.shape[2:] != spectral_x.shape[2:]:
            raise ValueError("Spatial and spectral views must share batch and spatial shape")
        if spatial_x.shape[1] != self.spatial_channels:
            raise ValueError(f"Expected {self.spatial_channels} PCA channels")
        if spectral_x.shape[1] != self.spectral_channels:
            raise ValueError(f"Expected {self.spectral_channels} native bands")

        net = self.upstream
        b, _, height, width = spatial_x.shape
        image_features = net.spat_encoder(spatial_x)
        spectral_feature = net.spec_encoder(spectral_x)[0]
        spectral_feature = net.pool(spectral_feature).view(b, -1)
        weights = (
            net.fc_spec1(spectral_feature).view(b, -1, 1, 1),
            net.fc_spec2(spectral_feature).view(b, -1, 1, 1),
            net.fc_spec3(spectral_feature).view(b, -1, 1, 1),
            net.fc_spec4(spectral_feature).view(b, -1, 1, 1),
        )
        reduced = [
            reducer((1.0 + weight) * feature)
            for reducer, weight, feature in zip(
                (net.DR1, net.DR2, net.DR3, net.DR4), weights, image_features
            )
        ]
        logits = net.cls(net.conv(torch.cat(reduced, dim=1)))
        if logits.shape[-2:] != (height, width):
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
        return logits

    def encoder_depth(self, name: str) -> int | None:
        marker = ".blocks."
        if marker not in name:
            return None
        try:
            return int(name.split(marker, 1)[1].split(".", 1)[0])
        except ValueError:
            return None
