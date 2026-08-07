"""Thin adapter around the author-released HybridSN notebook implementation."""

from __future__ import annotations

import ast
import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_NOTEBOOK = PROJECT_ROOT / "third_party" / "HybridSN" / "HybridSN.ipynb"


@lru_cache(maxsize=1)
def _upstream_hybridsn_class() -> type[nn.Module]:
    """Load only the upstream `class HybridSN` AST from its released notebook."""
    notebook = json.loads(UPSTREAM_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "class HybridSN(nn.Module)" not in source:
            continue
        tree = ast.parse(source)
        class_nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "HybridSN"
        ]
        if len(class_nodes) != 1:
            raise RuntimeError("Could not isolate the upstream HybridSN class")
        module = ast.Module(body=class_nodes, type_ignores=[])
        namespace = {"torch": torch, "nn": nn}
        exec(compile(module, str(UPSTREAM_NOTEBOOK), "exec"), namespace)
        return namespace["HybridSN"]
    raise RuntimeError(f"HybridSN class not found in {UPSTREAM_NOTEBOOK}")


class HybridSN(nn.Module):
    """Interface adapter; all learned layers come from the upstream class."""

    def __init__(self, in_channels: int = 30, patch_size: int = 25, num_classes: int = 17):
        super().__init__()
        upstream_class = _upstream_hybridsn_class()
        self.upstream = upstream_class(in_channels, patch_size, num_classes)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes

    @property
    def halo(self) -> int:
        return self.patch_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upstream(x)

    def forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Exact fully-convolutional evaluation of the frozen upstream layer graph."""
        model = self.upstream
        x = model.conv1(x.unsqueeze(1))
        x = model.conv2(x)
        x = model.conv3(x)
        batch, channels, depth, height, width = x.shape
        x = x.reshape(batch, channels * depth, height, width)
        x = model.conv4(x)

        linear1 = model.dense1[0]
        linear2 = model.dense2[0]
        linear3 = model.dense3[0]
        kernel = self.patch_size - 8
        x = F.relu(
            F.conv2d(
                x,
                linear1.weight.reshape(linear1.out_features, 64, kernel, kernel),
                linear1.bias,
            ),
            inplace=True,
        )
        x = F.relu(
            F.conv2d(x, linear2.weight[:, :, None, None], linear2.bias), inplace=True
        )
        return F.conv2d(x, linear3.weight[:, :, None, None], linear3.bias)

