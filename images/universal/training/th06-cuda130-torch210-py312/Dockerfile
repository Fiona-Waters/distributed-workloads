# Universal Image Dockerfile
#
# FIPS-friendly Features:
# - Build tools are isolated in intermediate stage
# - Final image contains only runtime dependencies
# - OpenSSL FIPS mode supported via base image
# - Uses pip install (additive) to preserve base image packages
#
# Build Modes:
# - Midstream (default): DOWNSTREAM=false - installs system packages and sets env vars
# - Downstream: DOWNSTREAM=true - skips midstream-only sections (base image has them)

################################################################################
# Build Arguments
################################################################################
ARG BASE_IMAGE=quay.io/opendatahub/odh-workbench-jupyter-minimal-cuda-py312-ubi9:3.4-v1.43
ARG PYTHON_VERSION=3.12
ARG DOWNSTREAM=false

################################################################################
# Build Stage - Install Python Dependencies
################################################################################
FROM ${BASE_IMAGE} AS builder

USER 0
WORKDIR /tmp/deps

# Copy requirements files
COPY --chown=1001:0 pyproject.toml requirements.txt ./

# Switch to user 1001 for pip installations
USER 1001
WORKDIR /opt/app-root/src

# Install dependencies from AIPCC 3.5 index
RUN echo "tilelang==0.1.9" > /tmp/deps/overrides.txt && \
    uv pip install --no-cache-dir \
    --index-strategy=unsafe-best-match \
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.5/cuda13.0-ubi9/simple/ \
    --override /tmp/deps/overrides.txt \
    -r /tmp/deps/requirements.txt

# opencv-python-headless is pulled by vllm + mistral-common but only needed for
# the optional OpenCV video backend (PyAV is the default since vllm#39986).
# Its cv2/typing/__init__.py triggers `import cv2` during pkgutil scanning in
# multiprocessing.spawn child processes, which fails with
# "libavcodec.so.60: cannot open shared object file" on UBI9.
# Upstream vLLM is making it optional: https://github.com/vllm-project/vllm/issues/40741
RUN uv pip uninstall opencv-python-headless 2>/dev/null || true

# Remove tilelang's libcudart stub which shadows the real CUDA runtime in vLLM subprocesses
RUN rm -f /opt/app-root/lib/python3.12/site-packages/tilelang/lib/libcudart_stub.so \
 && rm -f /opt/app-root/lib64/python3.12/site-packages/tilelang/lib/libcudart_stub.so

# TEMPORARY: kernels>=0.15 requires version= in LayerRepository but transformers 5.12.1
# doesn't provide it yet (huggingface/transformers#46291, fixed in PR #46293).
# Downgrade to 0.14.1 which doesn't enforce this.
RUN uv pip install --no-cache-dir \
    --index-url=https://pypi.org/simple/ \
    "kernels==0.14.1" "kernels-data==0.14.1"

# TEMPORARY: training-hub from fork branch to test vLLM 0.21 subprocess exit fix
RUN uv pip install --no-cache-dir --no-deps \
    --index-url=https://pypi.org/simple/ \
    "training-hub @ git+https://github.com/Fiona-Waters/training_hub.git@vllm-fix"

# TEMPORARY: openpipe-art not yet in AIPCC (AIPCC-16172)
# Remove this step and add to pyproject.toml once AIPCC packages it
# --no-deps: weave (transitive) pulls polyfile-weave which registers cv2.typing
# stubs that crash under multiprocessing.spawn (libavcodec.so.60 not in UBI9).
# Explicitly install only the runtime deps ART needs for the training-hub path.
RUN uv pip install --no-cache-dir --no-deps \
    --index-url=https://pypi.org/simple/ \
    openpipe-art==0.5.17
RUN uv pip install --no-cache-dir \
    --index-url=https://pypi.org/simple/ \
    "polars>=1.26.0" "tblib>=3.0.0" "litellm>=1.71.1"

# TEMPORARY: unsloth/unsloth-zoo not yet in AIPCC 3.5 (being re-added)
# Using opendatahub-io midstream forks with trl 1.x / datasets 5.x compat patches
# --no-deps: all transitive deps (torch, triton, peft, trl, etc.) already installed above
RUN uv pip install --no-cache-dir --no-deps \
    "unsloth-zoo @ git+https://github.com/opendatahub-io/unsloth-zoo.git@v2026.6.7+rhaiv.0" \
    "unsloth @ git+https://github.com/opendatahub-io/unsloth.git@v2026.6.9+rhaiv.0"

# Fix permissions for OpenShift
ARG PYTHON_VERSION
USER 0
RUN chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages \
 && fix-permissions /opt/app-root -P

# Clean up
RUN rm -rf /tmp/deps

################################################################################
# Final Stage - FIPS-friendly Runtime
################################################################################
FROM ${BASE_IMAGE} AS final

LABEL name="universal:th06-cuda130-py312-torch211" \
      summary="TH06 CUDA 13.0 Python 3.12 image with PyTorch 2.11.0" \
      description="Universal image combining minimal Jupyter workbench and runtime ML stack (CUDA 13.0, PyTorch 2.11.0, FlashAttention 2.8.3, vLLM 0.21.0) on UBI9" \
      io.k8s.display-name="TH06 CUDA 13.0 Python 3.12 (Workbench + Runtime)" \
      io.k8s.description="Universal image: Jupyter workbench by default; runtime when command provided."

USER 0
WORKDIR /opt/app-root/src

################################################################################
# MIDSTREAM ONLY: Environment variables and system packages
# Controlled by ARG DOWNSTREAM (default: false)
# - DOWNSTREAM=false (midstream): Installs CUDA dev tools, RDMA packages, sets env vars
# - DOWNSTREAM=true: Skips this section (AIPCC base image has everything pre-configured)
################################################################################
ARG DOWNSTREAM

# Environment variables for NVIDIA and CUDA
# These are safe to set in both modes - they define standard paths
# In downstream, the base image may override these appropriately
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    CPATH=/usr/local/cuda/include:${CPATH} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# System packages (MIDSTREAM ONLY - skipped when DOWNSTREAM=true)
# Repo files for CUDA and RDMA packages
COPY cuda.repo mellanox.repo /etc/yum.repos.d/

RUN if [ "${DOWNSTREAM}" != "true" ]; then \
    echo "MIDSTREAM BUILD: Installing system packages..." && \
    dnf install -y --setopt=install_weak_deps=False \
        perl \
        mesa-libGL \
        skopeo \
        libibverbs-utils \
        infiniband-diags \
        libibumad \
        librdmacm \
        librdmacm-utils \
        rdma-core \
        protobuf \
        cuda-cudart-devel-13-0 \
        cuda-nvcc-13-0 && \
    dnf clean all && rm -rf /var/cache/dnf/*; \
else \
    echo "DOWNSTREAM BUILD: Skipping system packages (provided by base image)"; \
fi
################################################################################
# END MIDSTREAM ONLY
################################################################################

# Copy Python site-packages and CLI entry points from builder stage
# This excludes any build artifacts (FIPS friendly)
ARG PYTHON_VERSION
COPY --from=builder /opt/app-root/lib/python${PYTHON_VERSION}/site-packages /opt/app-root/lib/python${PYTHON_VERSION}/site-packages
COPY --from=builder /opt/app-root/bin /opt/app-root/bin

# Remove uv from final image (inherited from base image, not needed at runtime)
RUN rm -f /opt/app-root/bin/uv

# Copy license file
COPY LICENSE.md /licenses/cuda-license.md

# Copy entrypoint
COPY --chmod=0755 entrypoint-universal.sh /usr/local/bin/entrypoint-universal.sh

# Fix permissions for OpenShift (final stage)
RUN fix-permissions /opt/app-root -P \
 && chmod -R g+w /opt/app-root/lib/python${PYTHON_VERSION}/site-packages

USER 1001
WORKDIR /opt/app-root/src

ENTRYPOINT ["/usr/local/bin/entrypoint-universal.sh"]
CMD ["start-notebook.sh"]
