#!/bin/bash
# =============================================================================
# RK3588 CPU-Only Stress Test Launcher
#
# Usage:
#   sudo ./run_cpu_stress_test.sh [duration_minutes]
#
# Examples:
#   sudo ./run_cpu_stress_test.sh 60
# =============================================================================

DURATION_MIN=${1:-30}
DURATION_SEC=$(( DURATION_MIN * 60 ))
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/bear-rk3588-matmul-bench/rk3588_stress_cpu_${TIMESTAMP}"
MATMUL_DIR="/bear-rk3588-matmul-bench"

# CPU stress parameters
CPU_CORES=8
CPU_LOAD=50
VM_WORKERS=2
VM_BYTES="1G"

LOG_INTERVAL=1

PID_LOGGER=""
PID_CPU=""

# =============================================================================
mkdir -p "$LOG_DIR"

echo "=============================================="
echo " RK3588 CPU Stress Test"
echo "=============================================="
echo " Duration   : ${DURATION_MIN} min (${DURATION_SEC}s)"
echo " Output dir : ${LOG_DIR}"
echo " CPU stress : ${CPU_CORES} cores @ ${CPU_LOAD}% + ${VM_WORKERS}x${VM_BYTES} VM"
echo "=============================================="
echo ""

# --- Pre-test snapshot ---
echo "# Pre-test system state" > "${LOG_DIR}/system_info.txt"
{
    echo "Date: $(date -Iseconds)"
    echo "Kernel: $(uname -r)"
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
    [[ -n "$PID_CPU" ]]    && kill "$PID_CPU"    2>/dev/null
    [[ -n "$PID_LOGGER" ]] && kill "$PID_LOGGER" 2>/dev/null
    wait 2>/dev/null
    echo "[Done] Logs saved to: ${LOG_DIR}/"
    ls -1 "${LOG_DIR}/"
    echo ""
    echo "Analyze with:"
    echo "  python3 plot_cpu_thermal.py ${LOG_DIR}/thermal_log.csv --save ${LOG_DIR}/report.png"
}
trap cleanup EXIT INT TERM

# --- 1) Thermal Logger ---
echo "[1] Starting thermal logger (interval=${LOG_INTERVAL}s)..."
bash "${MATMUL_DIR}/thermal_logger_cpu.sh" \
    "$LOG_INTERVAL" "$DURATION_SEC" \
    "${LOG_DIR}/thermal_log.csv" &
PID_LOGGER=$!
sleep 1  # 로거가 헤더를 쓸 시간 확보

# --- 2) CPU Stress ---
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

echo ""
echo "All processes started. Running for ${DURATION_MIN} min..."
[[ -n "$PID_LOGGER" ]] && echo "  Logger  PID: ${PID_LOGGER}"
[[ -n "$PID_CPU" ]]    && echo "  CPU     PID: ${PID_CPU}"
echo ""

echo "Waiting ${DURATION_SEC}s... (Ctrl+C to abort early)"
sleep "$DURATION_SEC"

echo "Duration complete. Shutting down..."