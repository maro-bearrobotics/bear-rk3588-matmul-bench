#!/bin/bash
# =============================================================================
# RK3588 CPU/NPU Thermal & Utilization Logger
# Logs temperature, CPU frequency, CPU utilization, NPU load to CSV
#
# Usage:
#   sudo ./thermal_logger.sh [interval_sec] [duration_sec] [output_file]
#   sudo ./thermal_logger.sh 1 3600 /tmp/thermal_log.csv
# =============================================================================

INTERVAL=${1:-1}
DURATION=${2:-3600}
OUTFILE=${3:-"/tmp/thermal_log_$(date +%Y%m%d_%H%M%S).csv"}

# --- RK3588 Thermal Zone Discovery ---
discover_zones() {
    echo "# Thermal zone mapping on this device:" >&2
    for z in /sys/class/thermal/thermal_zone*/type; do
        zid=$(echo "$z" | grep -oP 'zone\K[0-9]+')
        ztype=$(cat "$z" 2>/dev/null)
        echo "#   zone${zid}: ${ztype}" >&2
    done
    echo "#" >&2
}

# --- Read helpers ---
read_temp() {
    local raw=$(cat "/sys/class/thermal/thermal_zone${1}/temp" 2>/dev/null)
    if [[ -n "$raw" ]]; then
        echo "scale=1; $raw / 1000" | bc
    else
        echo "N/A"
    fi
}

read_cpu_freq() {
    local raw=$(cat "/sys/devices/system/cpu/cpu${1}/cpufreq/scaling_cur_freq" 2>/dev/null)
    if [[ -n "$raw" ]]; then
        echo "scale=0; $raw / 1000" | bc
    else
        echo "N/A"
    fi
}

read_cpu_governor() {
    cat "/sys/devices/system/cpu/cpu${1}/cpufreq/scaling_governor" 2>/dev/null || echo "N/A"
}

read_npu_load_parsed() {
    local raw=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null)
    if [[ -n "$raw" ]]; then
        local c0=$(echo "$raw" | grep -oP 'Core0:\s*\K[0-9]+')
        local c1=$(echo "$raw" | grep -oP 'Core1:\s*\K[0-9]+')
        local c2=$(echo "$raw" | grep -oP 'Core2:\s*\K[0-9]+')
        echo "${c0:-0},${c1:-0},${c2:-0}"
    else
        echo "0,0,0"
    fi
}

read_npu_load() {
    local raw=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null)
    if [[ -n "$raw" ]]; then
        echo "$raw" | grep -oP '[0-9]+%' | tr '\n' ',' | sed 's/,$//'
    else
        echo "N/A,N/A,N/A"
    fi
}

# --- CPU Utilization from /proc/stat ---
# Arrays to hold previous idle/total per CPU
declare -A PREV_IDLE PREV_TOTAL

init_cpu_stats() {
    while IFS=' ' read -r label u n s idle iow irq sirq st rest; do
        if [[ "$label" =~ ^cpu[0-9]+$ ]]; then
            local idx=${label#cpu}
            local total=$((u + n + s + idle + iow + irq + sirq + st))
            PREV_IDLE[$idx]=$idle
            PREV_TOTAL[$idx]=$total
        fi
    done < /proc/stat
}

read_cpu_util() {
    # Compute per-core CPU% since last call, output: "u0,u1,...,u7"
    local result=""
    while IFS=' ' read -r label u n s idle iow irq sirq st rest; do
        if [[ "$label" =~ ^cpu[0-9]+$ ]]; then
            local idx=${label#cpu}
            local total=$((u + n + s + idle + iow + irq + sirq + st))
            local prev_idle=${PREV_IDLE[$idx]:-0}
            local prev_total=${PREV_TOTAL[$idx]:-0}
            local d_total=$((total - prev_total))
            local d_idle=$((idle - prev_idle))
            local util=0
            if (( d_total > 0 )); then
                util=$(( (d_total - d_idle) * 100 / d_total ))
            fi
            PREV_IDLE[$idx]=$idle
            PREV_TOTAL[$idx]=$total
            if [[ -n "$result" ]]; then
                result="${result},${util}"
            else
                result="${util}"
            fi
        fi
    done < /proc/stat
    echo "$result"
}

# --- Pre-flight checks ---
if [[ $EUID -ne 0 ]]; then
    echo "WARNING: Not running as root. NPU load may not be readable." >&2
fi

discover_zones

N_ZONES=$(ls -d /sys/class/thermal/thermal_zone*/ 2>/dev/null | wc -l)

echo "# ============================================" >&2
echo "# RK3588 Thermal Logger" >&2
echo "# Interval: ${INTERVAL}s | Duration: ${DURATION}s" >&2
echo "# Output:   ${OUTFILE}" >&2
echo "# Thermal zones detected: ${N_ZONES}" >&2
echo "# CPU governors:" >&2
for c in 0 4 6; do
    echo "#   CPU${c}: $(read_cpu_governor $c)" >&2
done
echo "# ============================================" >&2
echo "# Started at: $(date -Iseconds)" >&2
echo "#" >&2

# --- CSV Header ---
HEADER="timestamp,elapsed_sec"
for ((z=0; z<N_ZONES; z++)); do
    ztype=$(cat "/sys/class/thermal/thermal_zone${z}/type" 2>/dev/null | tr '-' '_')
    HEADER="${HEADER},temp_${ztype}_C"
done
for c in 0 1 2 3 4 5 6 7; do
    HEADER="${HEADER},cpu${c}_freq_mhz"
done
for c in 0 1 2 3 4 5 6 7; do
    HEADER="${HEADER},cpu${c}_util_pct"
done
HEADER="${HEADER},npu_core0_pct,npu_core1_pct,npu_core2_pct"

echo "$HEADER" > "$OUTFILE"

# --- Init CPU stats baseline ---
init_cpu_stats
sleep 0.1

# --- Main loop ---
START=$(date +%s%N)
END_TIME=$(( $(date +%s) + DURATION ))
COUNT=0

echo "Logging started. Press Ctrl+C to stop early." >&2

trap 'echo ""; echo "Stopped after ${COUNT} samples. Output: ${OUTFILE}" >&2; exit 0' INT TERM

while [[ $(date +%s) -lt $END_TIME ]]; do
    NOW=$(date +%s%N)
    ELAPSED=$(echo "scale=1; ($NOW - $START) / 1000000000" | bc)
    TS=$(date -Iseconds)

    ROW="${TS},${ELAPSED}"

    # Temperatures
    for ((z=0; z<N_ZONES; z++)); do
        ROW="${ROW},$(read_temp $z)"
    done

    # CPU frequencies
    for c in 0 1 2 3 4 5 6 7; do
        ROW="${ROW},$(read_cpu_freq $c)"
    done

    # CPU utilization
    ROW="${ROW},$(read_cpu_util)"

    # NPU load
    ROW="${ROW},$(read_npu_load_parsed)"

    echo "$ROW" >> "$OUTFILE"
    COUNT=$((COUNT + 1))

    # Progress every 30 samples
    if (( COUNT % 30 == 0 )); then
        echo "[${ELAPSED}s] samples=${COUNT} | NPU: $(read_npu_load) $(read_temp 6)°C | A76-0: $(read_temp 1)°C @ $(read_cpu_freq 4)MHz | A76-1: $(read_temp 2)°C @ $(read_cpu_freq 6)MHz | A55: $(read_temp 3)°C | SoC: $(read_temp 0)°C" >&2
    fi

    sleep "$INTERVAL"
done

echo "" >&2
echo "Done. ${COUNT} samples written to ${OUTFILE}" >&2
echo "Analyze with: python3 plot_thermal.py ${OUTFILE}" >&2