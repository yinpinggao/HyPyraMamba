# M4 MambaHSI source and runtime audit

## Author source

- Repository: <https://github.com/li-yapeng/MambaHSI>
- Pinned checkout: `a705284aef14e802bb1a09b2a5e73abaefc0b2d0`
- Imported class: `third_party/MambaHSI/model/MambaHSI.py::MambaHSI`
- The upstream checkout was clean after the adapter work. No author network
  layer was copied or edited.
- The release constructs `Mamba` with `d_state=16`, `d_conv=4`, `expand=2`,
  uses `hidden_dim=128` in its training entry point, and upsamples stride-4
  logits with bilinear interpolation and `align_corners=True` for loss/eval.

## Machine and active environments (2026-07-27)

- GPUs: four NVIDIA A100-SXM4-80GB plus two NVIDIA A100-SXM4-40GB.
- Driver: `580.173.02`; all six devices report compute capability 8.0.
- Shell/base Python: Python 3.11, `torch 2.6.0+cu124`, CUDA 12.4,
  `torch._C._GLIBCXX_USE_CXX11_ABI=False`. Neither `mamba_ssm` nor
  `causal_conv1d` is installed there.
- Project environment `/home/guest/anaconda3/envs/gyp_hsi_env`: Python 3.11,
  `torch 2.10.0+cu130`, CUDA 13.0, CXX11 ABI `True`, `mamba-ssm 2.3.1`,
  `causal-conv1d 1.6.1`, and Triton 3.6.0. Both CUDA extensions import.
- A direct CUDA smoke on GPU 0 passed: `[1,98,16,16]` produced native author
  logits `[1,17,4,4]` and adapter-resized logits `[1,17,16,16]`.

The base environment must not receive an arbitrary wheel. Its precompiled
wheel compatibility markers are exactly `cu12`, `torch2.6`, and
`cxx11abiFALSE` (with the platform tag `cp311-cp311-linux_x86_64`). For the
project environment, the corresponding three markers are `cu13`, `torch2.10`,
and `cxx11abiTRUE`. A wheel must match all three; if no such published wheel
exists, create a dedicated supported environment instead of compiling during a
training launch.

## Competition tile contract

- Input is one standardized, non-PCA tile shaped `[1,98,H,W]`.
- Output has 17 class channels. Native author logits have spatial stride four;
  the adapter optionally restores them to `H x W` using the author's own
  interpolation convention.
- Batch size is fixed to one because the released spatial branch reshapes the
  tensor to a single sequence of length `B*H*W`; batching independent tiles
  would let them interact. Multi-GPU parallelism should therefore assign
  independent seeds/models to GPUs.
- By explicit user choice, `configs/m4.yaml` uses 512-pixel tiles, 32-pixel
  overlap, and the author's
  `hidden_dim=128`, `mamba_type=both`, `token_num=4` setup.
- A strict-deterministic forward/backward/optimizer smoke with a real
  `[1,98,512,512]` training tile passed and peaked at 5.71 GiB allocated GPU
  memory. Dense CE selects only labeled pixels into `[N,17]` before loss so it
  does not invoke CUDA's non-deterministic `nll_loss2d` kernel.
- FP16 was rejected after the first four-rank epoch produced `NaN` training
  loss. With no other variable changed, BF16 passed five consecutive real-tile
  updates (`2.8625 -> 2.7448`) and the restarted formal epoch 1 was finite:
  loss `2.741818`, validation OA `0.123996`. A100 BF16 is therefore the recorded
  M4 mixed-precision setting.
