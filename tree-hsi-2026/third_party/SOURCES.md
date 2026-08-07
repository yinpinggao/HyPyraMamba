# Upstream model sources

The model architectures in this project are not local reimplementations. Each
formal model must be instantiated from the corresponding author repository;
`src/models/` may only contain compatibility adapters.

| ID | Author repository | pinned commit | local source | pretrained weight status |
|---|---|---|---|---|
| M1 HybridSN | <https://github.com/Pancakerr/HybridSN> | `01b3ba919d1b3d8d15b132ff3e804ea0488ca08f` | `HybridSN/HybridSN.ipynb` | no competition-compatible weight published |
| M2 SSFTT | <https://github.com/zgr6010/HSI_SSFTT> | `994f5d72e4cf22e92d9b5ab9743e33dfdf642e1d` | `HSI_SSFTT/cls_SSFTT_IP/SSFTTnet.py` | no author checkpoint published in the repository |
| M3 GAHT | <https://github.com/MeiShaohui/Group-Aware-Hierarchical-Transformer> | `d9340ded8fca9cca650a0bbae3853ae499aa638d` | `GAHT/models/` | README documents checkpoint paths, but no checkpoint files are present |
| M4 MambaHSI | <https://github.com/li-yapeng/MambaHSI> | `a705284aef14e802bb1a09b2a5e73abaefc0b2d0` | `MambaHSI/model/MambaHSI.py` | no author checkpoint published in the repository |
| M5 HyperSIGMA | <https://github.com/WHU-Sigma/HyperSIGMA> | `07e9ea24e3072fcb5c3a92a2bcb8185e43b295b9` | `HyperSIGMA/ImageClassification/` plus `Pretrain/` | official Spatial and Spectral ViT-B weights downloaded and verified |

Licenses are retained exactly where the upstream repository provides them.
Absence of an explicit license will be recorded before redistribution or
packaging; it does not block local experiment reproduction.

Downloaded official weight:

- `weights/hypersigma/spec-vit-base-ultra-checkpoint-1599.pth`
- SHA-256: `662845ed9669c9fc8f39b18b19a10dff61792a9c8700b6bfea439815f7c86d6f`
- `weights/hypersigma/spat-vit-base-ultra-checkpoint-1599.pth`
- SHA-256: `df6584e8124a9916687a21193bad0516439c97ed712ba7a854e1843d3d5fabc7`
