#!/usr/bin/env bash
# Remote workflow: TheRock gfx1153 nightly + rocprof-compute full CTest (PR B branch)
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/therock-gfx1153-rocprof-compute-build}"
REPO="${REPO:-$HOME/rocm-systems}"
ROCM_VERSION="${ROCM_VERSION:-7.15.0a20260712}"
GFX_ARCH="${GFX_ARCH:-gfx1153}"
USER_TAG="${USER_TAG:-rocprof}"
BRANCH="${BRANCH:-users/feizheng10/gfx1153-enable}"
IMAGE="rocprofiler-compute-therock-gfx1153-${USER_TAG}"
CONTAINER="${CONTAINER:-rocprof-compute-gfx1153-nightly}"
LOG_DIR="${WORKDIR}/reports"
LOG_FILE="${LOG_DIR}/workflow.log"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${WORKDIR}/docker" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== rocprof-compute gfx1153 TheRock nightly CTest workflow ==="
echo "Started: ${TIMESTAMP}"
echo "Host: $(hostname)"
echo "WORKDIR=${WORKDIR}"
echo "REPO=${REPO}"
echo "BRANCH=${BRANCH}"
echo "ROCM_VERSION=${ROCM_VERSION} GFX_ARCH=${GFX_ARCH}"

# --- Step 0: host GPU sanity ---
echo "--- Step 0: host GPU check ---"
rocminfo 2>/dev/null | head -30 || true
ls -l /dev/kfd /dev/dri 2>/dev/null || true

# --- Step 1: checkout PR B branch ---
echo "--- Step 1: checkout ${BRANCH} ---"
if [[ ! -d "${REPO}/.git" ]]; then
  git clone --filter=blob:none --sparse git@github.com:ROCm/rocm-systems.git "${REPO}"
  cd "${REPO}"
  git sparse-checkout set projects/rocprofiler-compute
else
  cd "${REPO}"
fi
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}" || true
git submodule update --init --recursive projects/rocprofiler-compute/src || \
  git submodule update --init --recursive
COMMIT="$(git -C projects/rocprofiler-compute rev-parse --short HEAD 2>/dev/null || git rev-parse --short HEAD)"
echo "Checked out commit: ${COMMIT}"
echo "${COMMIT}" > "${LOG_DIR}/git_commit.txt"

# --- Step 2: write persistent Docker files ---
echo "--- Step 2: create Docker files in ${WORKDIR}/docker ---"
cat > "${WORKDIR}/docker/Dockerfile.therock.gfx1153" <<DOCKERFILE
# TheRock nightly gfx1153 build for rocprof-compute testing
# Generated: ${TIMESTAMP}
FROM ubuntu:24.04

ARG ROCM_VERSION=${ROCM_VERSION}
ARG GFX_ARCH=${GFX_ARCH}

RUN apt-get update && apt-get install -y \\
    curl software-properties-common cmake locales git \\
    && add-apt-repository ppa:deadsnakes/ppa \\
    && apt-get update

RUN locale-gen en_US.UTF-8

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \\
    python3.12 python3.12-venv python3.12-dev python3-pip libsqlite3-dev \\
    build-essential pkg-config libnuma-dev libdw-dev libelf-dev \\
    autoconf libtool autotools-dev libpapi-dev libpfm4-dev libopenmpi-dev \\
    libudev-dev wget unzip

RUN rm -rf /rocm-venv && python3.12 -m venv /rocm-venv && \\
    /rocm-venv/bin/pip install --upgrade pip && \\
    /rocm-venv/bin/pip install \\
        --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \\
        "rocm[profiler,devel,libraries,device-\${GFX_ARCH}]==\${ROCM_VERSION}"

RUN /rocm-venv/bin/rocm-sdk init

ENV ROCM_PATH="/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel" \\
    HIP_PATH="/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel" \\
    HIP_DEVICE_LIB_PATH="/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/amdgcn/bitcode" \\
    PATH="/rocm-venv/bin:/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin:\${PATH}" \\
    LD_LIBRARY_PATH="/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib:/rocm-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib:\${LD_LIBRARY_PATH}"

COPY projects/rocprofiler-compute/requirements.txt /app/projects/rocprofiler-compute/requirements.txt
COPY projects/rocprofiler-compute/requirements-test.txt /app/projects/rocprofiler-compute/requirements-test.txt
COPY projects/rocprofiler-compute/requirements-development.txt /app/projects/rocprofiler-compute/requirements-development.txt
RUN /rocm-venv/bin/pip install \\
    -r /app/projects/rocprofiler-compute/requirements.txt \\
    -r /app/projects/rocprofiler-compute/requirements-test.txt \\
    -r /app/projects/rocprofiler-compute/requirements-development.txt

WORKDIR /app/projects/rocprofiler-compute
RUN git config --global --add safe.directory /app
CMD ["/bin/bash"]
DOCKERFILE

cat > "${WORKDIR}/docker/docker-compose.gfx1153.yml" <<COMPOSE
services:
  ${CONTAINER}:
    build:
      context: ${REPO}
      dockerfile: ${WORKDIR}/docker/Dockerfile.therock.gfx1153
      args:
        ROCM_VERSION: "${ROCM_VERSION}"
        GFX_ARCH: "${GFX_ARCH}"
    image: ${IMAGE}
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - "44"
      - "992"
    security_opt:
      - seccomp:unconfined
    volumes:
      - ${REPO}:/app
      - ${LOG_DIR}:/reports
    tty: true
    stdin_open: true
    cap_add:
      - SYS_PTRACE
COMPOSE

echo "Docker files written:"
ls -la "${WORKDIR}/docker/"

# --- Step 3: build Docker image ---
echo "--- Step 3: docker build (TheRock ${ROCM_VERSION} gfx1153) ---"
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image ${IMAGE} already exists; skipping docker build"
else
  docker compose -f "${WORKDIR}/docker/docker-compose.gfx1153.yml" build 2>&1 | tee "${LOG_DIR}/docker_build.log"
fi

# --- Step 4: build rocprof-compute + run full CTest ---
echo "--- Step 4: build and run full CTest suite ---"
docker compose -f "${WORKDIR}/docker/docker-compose.gfx1153.yml" run --rm "${CONTAINER}" bash -lc '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential pkg-config libnuma-dev libdw-dev libelf-dev \
  autoconf libtool autotools-dev libpapi-dev libpfm4-dev libopenmpi-dev libudev-dev wget unzip || true

cd /app/projects/rocprofiler-compute
echo "Inside container: $(hostname)"
/rocm-venv/bin/python --version
echo "ROCM_PATH=${ROCM_PATH}"
/rocm-venv/bin/rocminfo 2>/dev/null | head -20 || true

git submodule update --init --recursive src/ || true
/rocm-venv/bin/pip install -q -r requirements.txt -r requirements-test.txt

rm -rf build install
cmake -B build \
  -D CMAKE_INSTALL_PREFIX=install \
  -D ENABLE_TESTS=ON \
  -D INSTALL_TESTS=ON \
  -D ENABLE_COVERAGE=ON \
  -D PYTEST_NUMPROCS="$(nproc)" \
  -S .
cmake --build build --target install --parallel "$(nproc)" 2>&1 | tee /reports/cmake_build.log

cd build
echo "=== CTest: full suite ===" | tee /reports/ctest_full.log
set +e
ctest --output-on-failure -j"$(nproc)" -V 2>&1 | tee -a /reports/ctest_full.log
CTEST_EXIT=${PIPESTATUS[0]}
set -e
echo "${CTEST_EXIT}" > /reports/ctest_exit_code.txt
grep -E "tests passed|tests failed|tests not run|Total Test time" /reports/ctest_full.log | tee /reports/ctest_summary.txt || true

echo "CTest exit code: ${CTEST_EXIT}"
exit "${CTEST_EXIT}"
' 2>&1 | tee "${LOG_DIR}/container_run.log"

FINAL_EXIT=$?
echo "=== Workflow finished with exit code ${FINAL_EXIT} ===" | tee "${LOG_DIR}/workflow_status.txt"
echo "Reports in: ${LOG_DIR}"
ls -la "${LOG_DIR}/"
exit "${FINAL_EXIT}"
