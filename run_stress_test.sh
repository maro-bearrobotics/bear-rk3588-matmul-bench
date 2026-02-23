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
#   sudo ./run_stress_test.sh cpu 60     # CPU-only baseline
#   sudo ./run_stress_test.sh npu 60     # NPU-only baseline
#   sudo ./run_stress_test.sh both 60    # Combined interference test
# =============================================================================

set -e

MODE=${1:-both}
DURATION_MIN=${2:-30}
DURATION_SEC=$((DURATION_MIN * 60))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/bear/Works/bear-rk3588-matmul-bench/rk3588_stress_${MODE}_${TIMESTAMP}"
MATMUL_DIR="/home/bear/Works/bear-rk3588-matmul-bench"

# Validate mode
if [[ "$MODE" != "cpu" && "$MODE" != "npu" && "$MODE" != "both" ]]; then
    echo "ERROR: Invalid mode '${MODE}'. Use: cpu | npu | both" >&2
    exit 1
fi

# MatMul parameters (M K N)
MM_M=1024
MM_K=4096
MM_N=4096
MM_CORE=0  # 0 = auto (all 3 cores)

# CPU stress parameters
CPU_CORES=8
CPU_LOAD=50
VM_WORKERS=2
VM_BYTES="1G"

# Logging interval
LOG_INTERVAL=1

# =============================================================================
mkdir -p "$LOG_DIR"

echo "=============================================="
echo " RK3588 Stress Test  [mode: ${MODE}]"
echo "=============================================="
echo " Duration   : ${DURATION_MIN} min"
echo " Output dir : ${LOG_DIR}"
[[ "$MODE" == "cpu" || "$MODE" == "both" ]] && \
echo " CPU stress : ${CPU_CORES} cores @ ${CPU_LOAD}% + ${VM_WORKERS}x${VM_BYTES} VM"
[[ "$MODE" == "npu" || "$MODE" == "both" ]] && \
echo " NPU stress : MatMul ${MM_M}x${MM_K}x${MM_N}, core=${MM_CORE}"
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
        echo "  CPU${c}: governor=${gov}, max_freq=$((max/1000))MHz"
    done
    echo ""
    echo "--- Memory ---"
    free -h
} >> "${LOG_DIR}/system_info.txt"

cat "${LOG_DIR}/system_info.txt"
echo ""

# --- Cleanup on exit ---
cleanup() {
    echo ""
    echo "[Cleanup] Stopping all stress processes..."
    [[ -n "$PID_LOGGER" ]] && kill $PID_LOGGER 2>/dev/null
    [[ -n "$PID_CPU" ]]    && kill $PID_CPU 2>/dev/null
    [[ -n "$PID_NPU" ]]    && kill $PID_NPU 2>/dev/null
    wait 2>/dev/null
    echo "[Done] Logs saved to: ${LOG_DIR}/"
    ls -1 "${LOG_DIR}/"
    echo ""
    echo "Analyze with:"
    echo "  python3 plot_thermal.py ${LOG_DIR}/thermal_log.csv --save ${LOG_DIR}/report.png"
}
trap cleanup EXIT INT TERM

# --- 1) Start Thermal Logger ---
echo "[1] Starting thermal logger (interval=${LOG_INTERVAL}s)..."
bash thermal_logger.sh "$LOG_INTERVAL" "$DURATION_SEC" "${LOG_DIR}/thermal_log.csv" &
PID_LOGGER=$!
sleep 1

# --- 2) Start CPU Stress (if cpu or both) ---
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
    echo "[2] CPU stress: SKIPPED (mode=${MODE})"
fi

# --- 3) Start NPU Stress (if npu or both) ---
if [[ "$MODE" == "npu" || "$MODE" == "both" ]]; then
    echo "[3] Starting NPU stress (MatMul ${MM_M}x${MM_K}x${MM_N})..."
    if [[ -x "${MATMUL_DIR}/bench" ]]; then
        timeout "${DURATION_SEC}s" \
            taskset -c 4-7 "${MATMUL_DIR}/bench" \
            "$MM_M" "$MM_K" "$MM_N" "$MM_CORE" \
            > "${LOG_DIR}/npu_stress.log" 2>&1 &
        PID_NPU=$!
    else
        echo "ERROR: ${MATMUL_DIR}/bench not found!" >&2
        exit 1
    fi
else
    echo "[3] NPU stress: SKIPPED (mode=${MODE})"
fi

echo ""
echo "All processes started. Running for ${DURATION_MIN} minutes... [mode: ${MODE}]"
[[ -n "$PID_LOGGER" ]] && echo "  Logger PID: ${PID_LOGGER}"
[[ -n "$PID_CPU" ]]    && echo "  CPU PID:    ${PID_CPU}"
[[ -n "$PID_NPU" ]]    && echo "  NPU PID:    ${PID_NPU}"
echo ""

# Wait for workloads to finish
[[ -n "$PID_CPU" ]] && { wait $PID_CPU 2>/dev/null; echo "CPU stress completed."; }
[[ -n "$PID_NPU" ]] && { wait $PID_NPU 2>/dev/null; echo "NPU stress completed."; }
kill $PID_LOGGER 2>/dev/null
wait $PID_LOGGER 2>/dev/null
echo "Logger stopped."