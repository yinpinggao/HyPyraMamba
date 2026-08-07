FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git bzip2 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN curl -fsSL -o /tmp/miniconda.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-py310_24.5.0-0-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

WORKDIR /workspace/tree-hsi-2026
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy

# Required build arguments must point to release assets, never source archives.
# Expected wheel markers: CPython 3.10, cu12, torch2.2, cxx11abiTRUE, linux x86_64.
ARG CAUSAL_CONV1D_WHEEL_URL
ARG MAMBA_SSM_WHEEL_URL
ARG TORCH_WHEEL_URL
RUN test -n "${TORCH_WHEEL_URL}" \
    && test -n "${CAUSAL_CONV1D_WHEEL_URL}" \
    && test -n "${MAMBA_SSM_WHEEL_URL}" \
    && conda run -n tree-hsi-2026 pip install "${TORCH_WHEEL_URL}" \
    && conda run -n tree-hsi-2026 pip install --no-deps \
       "${CAUSAL_CONV1D_WHEEL_URL}" "${MAMBA_SSM_WHEEL_URL}" \
    && conda run -n tree-hsi-2026 python -c \
       "import torch,mamba_ssm,causal_conv1d; assert torch.__version__.startswith('2.2'); assert torch.version.cuda.startswith('12.'); assert torch._C._GLIBCXX_USE_CXX11_ABI"

COPY . /workspace/tree-hsi-2026
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tree-hsi-2026"]
CMD ["python", "src/train.py", "--help"]
