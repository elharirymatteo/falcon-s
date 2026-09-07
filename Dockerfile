# FALCON-S: the reproduction environment, pinned to the same lock the venv uses.
#
# The benchmark's plant is an NVIDIA Warp kernel set, so the image needs a CUDA runtime and the
# container needs `--gpus all`. Everything the paper is reproduced from — the 39 checkpoints, the
# aircraft data and the shipped results — is in the repository and therefore in the image.
#
#   docker build -t falcons:1.0.0 .
#   docker run --rm --gpus all falcons:1.0.0                     # falcons check
#   docker run --rm --gpus all falcons:1.0.0 pytest -q -m "not slow"
#   docker run --rm --gpus all -v "$PWD/out:/out" falcons:1.0.0 falcons reproduce --out /out
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /falcon-s

# The lock first, on its own layer: the dependency set changes far less often than the code, so a
# code edit does not re-download PyTorch.
COPY requirements.lock pyproject.toml ./
RUN python3.10 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.lock

COPY src src
COPY checkpoints checkpoints
COPY results results
COPY tests tests
COPY scripts scripts
COPY README.md PROVENANCE.md LICENSE ./
RUN .venv/bin/pip install --no-cache-dir -e ".[test]" --no-deps

ENV PATH=/falcon-s/.venv/bin:$PATH
CMD ["falcons", "check"]
