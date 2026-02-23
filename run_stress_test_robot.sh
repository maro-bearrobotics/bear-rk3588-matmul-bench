#!/bin/bash
# =============================================================================
# RK3588 Combined CPU + NPU Stress Test Launcher
#
# Usage:
#   sudo ./run_stress_test.sh [mode] [duration_minutes]
#
#   mode: cpu | npu | both (default: both)
#
# Examples:
#   sudo ./run_stress_test.sh cpu 60
#   sudo ./run_stress_test.sh npu 60
#   sudo ./run_stress_test.sh both 60
# =============================================================================

# set -e 제거: wait의 반환값이 non-zero여도 스크립트가 죽지 않도록
# (stress-ng / npu_stress가 비정상 종료해도 로거는 유지)

MODE=${1:-both}
DURATION_MIN=${2:-30}
DURATION_SEC=$(( DURATION_MIN * 60 ))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/bear-rk3588-matmul-bench/rk3588_stress_${MODE}_${TIMESTAMP}"
MATMUL_DIR="/bear-rk3588-matmul-bench"

# Validate mode
if [[ "$MODE" != "cpu" && "$MODE" != "npu" && "$MODE" != "both" ]]; then
    echo "ERROR: Invalid mode '${MODE}'. Use: cpu | npu | both" >&2
    exit 1
fi

# MatMul parameters
MM_M=1024
MM_K=4096
MM_N=4096
MM_TYPE=0  # 0=INT8, 1=FP16

# CPU stress parameters
CPU_CORES=8
CPU_LOAD=50
VM_WORKERS=2
VM_BYTES="1G"

LOG_INTERVAL=1

PID_LOGGER=""
PID_CPU=""
PID_NPU=""

# =============================================================================
mkdir -p "$LOG_DIR"

echo "=============================================="
echo " RK3588 Stress Test  [mode: ${MODE}]"
echo "=============================================="
echo " Duration   : ${DURATION_MIN} min (${DURATION_SEC}s)"
echo " Output dir : ${LOG_DIR}"
[[ "$MODE" == "cpu" || "$MODE" == "both" ]] && \
echo " CPU stress : ${CPU_CORES} cores @ ${CPU_LOAD}% + ${VM_WORKERS}x${VM_BYTES} VM"
[[ "$MODE" == "npu" || "$MODE" == "both" ]] && \
echo " NPU stress : npu_stress ${MM_M}x${MM_K}x${MM_N} type=${MM_TYPE}"
echo "=============================================="
echo ""

# --- Pre-test snapshot ---
echo "# Pre-test system state" > "${LOG_DIR}/system_info.txt"
{
    echo "Date: $(date -Iseconds)"
    echo "Kernel: $(uname -r)"
    echo "Mode: ${MODE}"
    echo ""
    echo "--- RKNPU Driver ---"
    cat /sys/kernel/debug/rknpu/version 2>/dev/null || echo "N/A"
    echo ""
    echo "--- Thermal Zone Mapping ---"
    for z in /sys/class/thermal/thermal_zone*/type; do
        zid=$(echo "$z" | grep -oP 'zone\K[0-9]+')
        echo "  zone${zid}: $(cat $z)"
    done
    echo ""
    echo "--- CPU Freq Policy ---"
    for c in 0 4 6; do
        gov=$(cat /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor 2>/dev/null)
        max=$(cat /sys/devices/system/cpu/cpu${c}/cpufreq/scaling_max_freq 2>/dev/null)
        echo "  CPU${c}: governor=${gov}, max_freq=$(( max / 1000 ))MHz"
    done
    echo ""
    echo "--- Memory ---"
    free -h
} >> "${LOG_DIR}/system_info.txt"

cat "${LOG_DIR}/system_info.txt"
echo ""

# --- Cleanup ---
cleanup() {
    echo ""
    echo "[Cleanup] Stopping all stress processes..."
    [[ -n "$PID_NPU" ]]    && kill "$PID_NPU"    2>/dev/null
    [[ -n "$PID_CPU" ]]    && kill "$PID_CPU"    2>/dev/null
    [[ -n "$PID_LOGGER" ]] && kill "$PID_LOGGER" 2>/dev/null
    wait 2>/dev/null
    echo "[Done] Logs saved to: ${LOG_DIR}/"
    ls -1 "${LOG_DIR}/"
    echo ""
    echo "Analyze with:"
    echo "  python3 plot_thermal.py ${LOG_DIR}/thermal_log.csv --save ${LOG_DIR}/report.png"
}
trap cleanup EXIT INT TERM

# --- 1) Thermal Logger ---
echo "[1] Starting thermal logger (interval=${LOG_INTERVAL}s)..."
bash "${MATMUL_DIR}/thermal_logger.sh" \
    "$LOG_INTERVAL" "$DURATION_SEC" \
    "${LOG_DIR}/thermal_log.csv" &
PID_LOGGER=$!
sleep 1  # 로거가 헤더를 쓸 시간 확보

# --- 2) CPU Stress ---
if [[ "$MODE" == "cpu" || "$MODE" == "both" ]]; then
    echo "[2] Starting CPU stress (${CPU_CORES} cores @ ${CPU_LOAD}%)..."
    stress-ng \
        --cpu "$CPU_CORES" \
        --cpu-load "$CPU_LOAD" \
        --vm "$VM_WORKERS" \
        --vm-bytes "$VM_BYTES" \
        --vm-keep \
        --timeout "${DURATION_SEC}s" \
        --metrics-brief \
        > "${LOG_DIR}/cpu_stress.log" 2>&1 &
    PID_CPU=$!
else
    echo "[2] CPU stress: SKIPPED"
fi

# --- 3) NPU Stress ---
# bench는 단발성 벤치마크이므로 지속 실행용 npu_stress 사용
if [[ "$MODE" == "npu" || "$MODE" == "both" ]]; then
    NPU_BIN="${MATMUL_DIR}/bench"
    if [[ ! -x "$NPU_BIN" ]]; then
        echo "ERROR: ${NPU_BIN} not found. Build first:" >&2
        echo "  g++ bench_robot.cpp -o bench -I/usr/include/rknn -lrknnrt -lpthread -O3 -std=c++17" >&2
        exit 1
    fi
    echo "[3] Starting NPU stress (${MM_M}x${MM_K}x${MM_N} type=${MM_TYPE})..."
    taskset -c 4-7 "$NPU_BIN" \
        "$MM_M" "$MM_K" "$MM_N" "$MM_TYPE" \
        > "${LOG_DIR}/npu_stress.log" 2>&1 &
    PID_NPU=$!
else
    echo "[3] NPU stress: SKIPPED"
fi

echo ""
echo "All processes started. Running for ${DURATION_MIN} min..."
[[ -n "$PID_LOGGER" ]] && echo "  Logger  PID: ${PID_LOGGER}"
[[ -n "$PID_CPU" ]]    && echo "  CPU     PID: ${PID_CPU}"
[[ -n "$PID_NPU" ]]    && echo "  NPU     PID: ${PID_NPU}"
echo ""

# --- 지정된 시간만큼 대기 후 종료 ---
# wait 대신 sleep 사용: 개별 프로세스가 죽어도 전체가 중단되지 않음
echo "Waiting ${DURATION_SEC}s... (Ctrl+C to abort early)"
sleep "$DURATION_SEC"

echo "Duration complete. Shutting down..."
# cleanup은 trap EXIT에서 자동 호출됨