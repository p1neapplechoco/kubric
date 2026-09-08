# --- GPU-capable Blender-in-Docker image for the velocity-intervention dataset.
#
# The stock kubricdockerhub images ship Blender 2.93 built from source WITHOUT
# CUDA/OptiX kernels (see docker/cycles_free_patch.txt), so `--gpus all` alone
# never makes Cycles render on the GPU. This image instead installs Blender 4.2
# LTS as the pip `bpy` module (which bundles the CUDA + OptiX kernels) on top of
# NVIDIA's CUDA runtime image, with the pinned Python 3.11 dependency set from
# requirements_render.txt.
#
# Build (from the repository root):
#   docker build -f docker/KubricGPU.Dockerfile -t kubric-gpu:4.2 .
#
# Run (requires the NVIDIA Container Toolkit on the host):
#   docker run --rm --gpus all \
#     --user $(id -u):$(id -g) \
#     --volume "$PWD:/kubric" --workdir /kubric \
#     kubric-gpu:4.2 \
#     python scripts/build_velocity_dataset.py --output output/velocity --count 4 --require-gpu

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    KUBRIC_USE_GPU=true \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Python 3.11 (deadsnakes) + the shared libraries the headless bpy wheel needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates curl gnupg git ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-venv python3.11-distutils \
      libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libglu1-mesa \
      libsm6 libice6 libxkbcommon0 libegl1 libgomp1 libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && python -m pip install --no-cache-dir --upgrade pip wheel setuptools

WORKDIR /kubric
COPY requirements_render.txt .
RUN python -m pip install --no-cache-dir -r requirements_render.txt

# Make the repository importable when mounted at /kubric (no `pip install -e .`
# needed, so the container works with any checkout mounted over this path).
ENV PYTHONPATH=/kubric

# Sanity check: bpy imports and reports its Cycles device types at build time.
RUN python -c "import bpy, numpy; print('bpy', bpy.app.version_string, 'numpy', numpy.__version__)"

CMD ["python", "-c", "import bpy; p=bpy.context.preferences.addons['cycles'].preferences; print({b: [d.name for d in p.get_devices_for_type(b)] for b in ('OPTIX','CUDA')})"]
