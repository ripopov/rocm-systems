#!/usr/bin/env bash
# gfx1153 mega_kernel-only metric health on gorgon-point-1 (Krackan2 / PCI 1002:1902)
set -euo pipefail

TAG="${TAG:-260714-gorgon}"
WORKDIR="${WORKDIR:-$HOME/therock-gfx1153-gorgon-build}"
REPO="${REPO:-$HOME/rocm-systems}"
COMPOSE="${WORKDIR}/docker/docker-compose.gfx1153.yml"
ROOT="${REPO}/projects/rocprofiler-compute"
LOG_DIR="${WORKDIR}/reports"
LOG_FILE="${LOG_DIR}/health_${TAG}.log"
REPORT="${ROOT}/gfx1153_metric_health_report_${TAG}.html"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo docker)
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== gfx1153 metric health (mega_kernel only) TAG=${TAG} $(date -u) ==="

if ! "${DOCKER[@]}" image inspect rocprofiler-compute-therock-gfx1153-gorgon >/dev/null 2>&1; then
  echo "--- docker build (image missing) ---"
  "${DOCKER[@]}" compose -f "${COMPOSE}" build
fi

"${DOCKER[@]}" compose -f "${COMPOSE}" run --rm rocprof-compute-gfx1153-nightly bash -lc "
set -euo pipefail
export HIP_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0 HSA_USE_SVM=0 HSA_XNACK=0
export PATH=/rocm-venv/bin:/rocm/bin:\${PATH:-}
export LD_LIBRARY_PATH=/rocm/lib:\${LD_LIBRARY_PATH:-}

cd /app/projects/rocprofiler-compute
echo '--- container GPU ---'
/rocm-venv/bin/rocminfo 2>/dev/null | head -25 || true

echo '--- build mega_kernel gfx1153 ---'
make -C sample/mega_kernel clean
make -C sample/mega_kernel mega_kernel_test_gfx1153 HIPCC=/rocm-venv/bin/hipcc
test -x sample/mega_kernel/mega_kernel_test_gfx1153

PROFILE='./src/rocprof-compute profile --no-roof --no-native-tool --overwrite'
WL_NAME=mega_kernel_${TAG}
LOG_DIR=workloads/logs_${TAG}
mkdir -p \"\${LOG_DIR}\"
rm -rf \"workloads/\${WL_NAME}\"

echo '--- profile mega_kernel ---'
set +e
\${PROFILE} -n \"\${WL_NAME}\" -- bash -lc './sample/mega_kernel/mega_kernel_test_gfx1153 -b 655360 -t 256; exit 0'
PROF_RC=\$?
set -e
if [[ \${PROF_RC} -ne 0 ]]; then
  echo \"WARN: profile returned \${PROF_RC} (continuing if workload dir exists)\"
fi

WL_BASE=workloads/\${WL_NAME}
if [[ ! -d \"\${WL_BASE}\" ]]; then
  echo \"ERROR: workload dir missing: \${WL_BASE}\" >&2
  ls -la workloads/ || true
  exit 1
fi

TARGET_DIR=\$(find \"\${WL_BASE}\" -mindepth 1 -maxdepth 1 -type d | head -1)
if [[ -z \"\${TARGET_DIR}\" ]]; then
  echo \"ERROR: no GPU model subdir under \${WL_BASE}\" >&2
  exit 1
fi
echo \"Using analyze path: \${TARGET_DIR}\"

echo '--- analyze mega_kernel (--view table) ---'
./src/rocprof-compute analyze -p \"\${TARGET_DIR}/\" -k 0 --view table > \"\${LOG_DIR}/\${WL_NAME}.log\" 2>&1
wc -l \"\${LOG_DIR}/\${WL_NAME}.log\"

echo '--- generate HTML report ---'
/rocm-venv/bin/python tools/generate_gfx1153_metric_health_report.py \\
  --out gfx1153_metric_health_report_${TAG}.html \\
  --date $(date +%Y-%m-%d) \\
  --host gorgon-point-1 \\
  --branch users/feizheng10/gfx1153-enable \\
  mega_kernel:\${LOG_DIR}/\${WL_NAME}.log

ls -la gfx1153_metric_health_report_${TAG}.html
echo HEALTH_DONE
"

echo "=== finished $(date -u) report=${REPORT} ==="
