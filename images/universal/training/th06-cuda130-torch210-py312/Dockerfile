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

# Install dependencies from AIPCC 3.5 EA1 index
RUN uv pip install --no-cache-dir \
    --index-strategy=unsafe-best-match \
    --index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.5-EA1/cuda13.0-ubi9-test/simple/ \
    -r /tmp/deps/requirements.txt

# opencv-python-headless is pulled by vllm + mistral-common but only needed for
# the optional OpenCV video backend (PyAV is the default since vllm#39986).
# Its cv2/typing/__init__.py triggers `import cv2` during pkgutil scanning in
# multiprocessing.spawn child processes, which fails with
# "libavcodec.so.60: cannot open shared object file" on UBI9.
# Upstream vLLM is making it optional: https://github.com/vllm-project/vllm/issues/40741
RUN uv pip uninstall opencv-python-headless 2>/dev/null || true

# training-hub from main (post-v0.8.0, includes PR #62 vLLM V1 LoRA fix)
# TODO: Replace with pinned AIPCC version once a new release is cut
RUN uv pip install --no-cache-dir --no-deps \
    "training-hub[lora,grpo] @ git+https://github.com/Red-Hat-AI-Innovation-Team/training_hub.git@main"

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

LABEL name="universal:th06-cuda130-py312-torch210" \
      summary="TH06 CUDA 13.0 Python 3.12 image with PyTorch 2.10.0" \
      description="Universal image combining minimal Jupyter workbench and runtime ML stack (CUDA 13.0, PyTorch 2.10.0, FlashAttention 2.8.3) on UBI9" \
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
