#!/bin/bash
# =============================================================================
# RK3588 CPU-Only Thermal & Utilization Logger
# - NPU load 컬럼 제거
# - cooling_device cur_state 유지 (CPU throttling 감지)
#
# Usage:
#   sudo ./thermal_logger_cpu.sh [interval_sec] [duration_sec] [output_file]
# =============================================================================

INTERVAL=${1:-1}
DURATION=${2:-3600}
OUTFILE=${3:-"/tmp/thermal_log_$(date +%Y%m%d_%H%M%S).csv"}

# =============================================================================
# Thermal Zone Discovery
# =============================================================================
discover_zones() {
    echo "# Thermal zone mapping on this device:" >&2
    for z in /sys/class/thermal/thermal_zone*/type; do
        zid=$(echo "$z" | grep -oP 'zone\K[0-9]+')
        ztype=$(cat "$z" 2>/dev/null)
        echo "#   zone${zid}: ${ztype}" >&2
    done
    echo "#" >&2
}

# =============================================================================
# Cooling Device Discovery
# =============================================================================
N_COOLING=0
declare -a COOLING_TYPES

discover_cooling() {
    echo "# Cooling device mapping on this device:" >&2
    local idx=0
    for cd_path in /sys/class/thermal/cooling_device*/; do
        local cdid
        cdid=$(basename "$cd_path" | grep -oP '\d+')
        local cdtype max_state
        cdtype=$(cat "${cd_path}type" 2>/dev/null || echo "unknown")
        max_state=$(cat "${cd_path}max_state" 2>/dev/null || echo "?")
        echo "#   cooling_device${cdid}: type=${cdtype}, max_state=${max_state}" >&2
        COOLING_TYPES[$idx]="${cdtype}"
        idx=$(( idx + 1 ))
    done
    N_COOLING=$idx
    echo "#" >&2
}

# --- Read helpers ---

read_temp() {
    local raw
    raw=$(cat "/sys/class/thermal/thermal_zone${1}/temp" 2>/dev/null)
    if [[ -n "$raw" ]]; then
        awk "BEGIN { printf \"%.1f\", $raw / 1000 }"
    else
        echo "N/A"
    fi
}

read_cpu_freq() {
    local raw
    raw=$(cat "/sys/devices/system/cpu/cpu${1}/cpufreq/scaling_cur_freq" 2>/dev/null)
    [[ -n "$raw" ]] && echo $(( raw / 1000 )) || echo "N/A"
}

read_cpu_governor() {
    cat "/sys/devices/system/cpu/cpu${1}/cpufreq/scaling_governor" 2>/dev/null || echo "N/A"
}

read_cooling_cur_state() {
    local val
    val=$(cat "/sys/class/thermal/cooling_device${1}/cur_state" 2>/dev/null)
    [[ -n "$val" ]] && echo "$val" || echo "N/A"
}

read_cooling_max_state() {
    local val
    val=$(cat "/sys/class/thermal/cooling_device${1}/max_state" 2>/dev/null)
    [[ -n "$val" ]] && echo "$val" || echo "0"
}

# --- CPU Utilization from /proc/stat ---
declare -A PREV_IDLE PREV_TOTAL

init_cpu_stats() {
    while IFS=' ' read -r label u n s idle iow irq sirq st rest; do
        if [[ "$label" =~ ^cpu[0-9]+$ ]]; then
            local idx=${label#cpu}
            local total=$(( u + n + s + idle + iow + irq + sirq + st ))
            PREV_IDLE[$idx]=$idle
            PREV_TOTAL[$idx]=$total
        fi
    done < /proc/stat
}

read_cpu_util() {
    local result=""
    while IFS=' ' read -r label u n s idle iow irq sirq st rest; do
        if [[ "$label" =~ ^cpu[0-9]+$ ]]; then
            local idx=${label#cpu}
            local total=$(( u + n + s + idle + iow + irq + sirq + st ))
            local prev_idle=${PREV_IDLE[$idx]:-0}
            local prev_total=${PREV_TOTAL[$idx]:-0}
            local d_total=$(( total - prev_total ))
            local d_idle=$(( idle - prev_idle ))
            local util=0
            (( d_total > 0 )) && util=$(( (d_total - d_idle) * 100 / d_total ))
            PREV_IDLE[$idx]=$idle
            PREV_TOTAL[$idx]=$total
            result="${result:+${result},}${util}"
        fi
    done < /proc/stat
    echo "$result"
}

# =============================================================================
# Pre-flight
# =============================================================================
[[ $EUID -ne 0 ]] && echo "WARNING: Not root. cooling_device may be unreadable." >&2

discover_zones
discover_cooling

N_ZONES=$(ls -d /sys/class/thermal/thermal_zone*/ 2>/dev/null | wc -l)

echo "# ============================================" >&2
echo "# RK3588 CPU Thermal Logger (NPU-free)" >&2
echo "# Interval: ${INTERVAL}s | Duration: ${DURATION}s" >&2
echo "# Output:   ${OUTFILE}" >&2
echo "# Thermal zones: ${N_ZONES} | Cooling devices: ${N_COOLING}" >&2
echo "# CPU governors:" >&2
for c in 0 4 6; do
    echo "#   CPU${c}: $(read_cpu_governor $c)" >&2
done
echo "# ============================================" >&2
echo "# Started at: $(date -Iseconds)" >&2
echo "#" >&2

# =============================================================================
# CSV Header
# =============================================================================
HEADER="timestamp,elapsed_sec"

# Temperatures (thermal zone 전체 유지 — NPU zone도 온도 참고용으로 기록)
for ((z=0; z<N_ZONES; z++)); do
    ztype=$(cat "/sys/class/thermal/thermal_zone${z}/type" 2>/dev/null | tr '-' '_')
    HEADER="${HEADER},temp_${ztype}_C"
done

# CPU frequencies
for c in 0 1 2 3 4 5 6 7; do
    HEADER="${HEADER},cpu${c}_freq_mhz"
done

# CPU utilization
for c in 0 1 2 3 4 5 6 7; do
    HEADER="${HEADER},cpu${c}_util_pct"
done

# ── NPU 컬럼 제거됨 ──

# Cooling device cur_state
for ((i=0; i<N_COOLING; i++)); do
    safe_type=$(echo "${COOLING_TYPES[$i]}" | tr '-' '_' | tr '/' '_')
    max_s=$(read_cooling_max_state $i)
    HEADER="${HEADER},cool${i}_${safe_type}_cur"
    echo "# cooling_device${i}: type=${COOLING_TYPES[$i]}, max_state=${max_s}" >&2
done
HEADER="${HEADER},any_throttle"

echo "$HEADER" > "$OUTFILE"

# =============================================================================
# Init & Main Loop
# =============================================================================
init_cpu_stats
sleep 0.1

START_SEC=$(date +%s)
END_TIME=$(( START_SEC + DURATION ))
COUNT=0

echo "Logging started. Press Ctrl+C to stop early." >&2

trap 'echo ""; echo "Stopped after ${COUNT} samples. Output: ${OUTFILE}" >&2; exit 0' INT TERM

while [[ $(date +%s) -lt $END_TIME ]]; do
    NOW_SEC=$(date +%s)
    ELAPSED=$(( NOW_SEC - START_SEC ))
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

    # ── NPU load 제거됨 ──

    # Cooling device cur_state
    ANY_THROTTLE=0
    for ((i=0; i<N_COOLING; i++)); do
        cur=$(read_cooling_cur_state $i)
        ROW="${ROW},${cur}"
        if [[ "$cur" =~ ^[0-9]+$ ]] && (( cur > 0 )); then
            ANY_THROTTLE=1
        fi
    done
    ROW="${ROW},${ANY_THROTTLE}"

    echo "$ROW" >> "$OUTFILE"
    COUNT=$(( COUNT + 1 ))

    # Progress every 30 samples
    if (( COUNT % 30 == 0 )); then
        THROTTLE_STATUS="OK"
        (( ANY_THROTTLE == 1 )) && THROTTLE_STATUS="⚠ THROTTLING"
        echo "[${ELAPSED}s] samples=${COUNT} | THROTTLE=${THROTTLE_STATUS} | A55: $(read_temp 0)°C @ $(read_cpu_freq 0)MHz | A76-0: $(read_temp 1)°C @ $(read_cpu_freq 4)MHz | A76-1: $(read_temp 2)°C @ $(read_cpu_freq 6)MHz" >&2
    fi

    sleep "$INTERVAL"
done

echo "" >&2
echo "Done. ${COUNT} samples written to ${OUTFILE}" >&2
echo "Analyze with: python3 plot_cpu_thermal.py ${OUTFILE}" >&2