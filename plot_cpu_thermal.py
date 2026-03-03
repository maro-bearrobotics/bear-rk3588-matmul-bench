#!/usr/bin/env python3
"""
RK3588 CPU-Only Thermal & Throttling Analysis Plotter
- NPU 패널 제거 (④ NPU Utilization, ⑥ NPU Temp vs Load)
- CPU 관련 패널만 유지: 온도 / CPU freq / CPU util / cooling_state

Usage: python3 plot_cpu_thermal.py /tmp/thermal_log_*.csv [--save output.png]
"""
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# =============================================================================
# Data Loading
# =============================================================================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment='#')
    df['elapsed_min'] = df['elapsed_sec'] / 60.0
    return df

# =============================================================================
# Cooling Device Column Helpers
# =============================================================================
def get_cooling_cols(df: pd.DataFrame) -> list[dict]:
    pattern = re.compile(r'^cool(\d+)_(.+)_cur$')
    result = []
    for col in df.columns:
        m = pattern.match(col)
        if m:
            result.append({
                'col':  col,
                'idx':  int(m.group(1)),
                'type': m.group(2),
            })
    return sorted(result, key=lambda x: x['idx'])

# =============================================================================
# Throttling Detection (freq drop + cur_state)
# =============================================================================
def detect_throttling(df: pd.DataFrame) -> dict:
    results = {}

    for c in range(8):
        col = f'cpu{c}_freq_mhz'
        if col not in df.columns:
            continue
        freq = pd.to_numeric(df[col], errors='coerce').dropna()
        if freq.empty:
            continue
        max_f  = freq.max()
        min_f  = freq.min()
        init_f = freq.head(10).mean()
        drop_pct = (1 - min_f / init_f) * 100 if init_f > 0 else 0
        results[c] = {
            'max_mhz': max_f, 'min_mhz': min_f,
            'init_mhz': init_f, 'drop_pct': drop_pct,
            'throttled_by_freq': drop_pct > 5,
            'throttled_by_cooling': False,
            'throttled': drop_pct > 5,
        }

    cooling_cols = get_cooling_cols(df)
    for meta in cooling_cols:
        vals = pd.to_numeric(df[meta['col']], errors='coerce').fillna(0)
        throttled_rows = (vals > 0).sum()
        if throttled_rows == 0:
            continue

        t = meta['type'].lower()
        if 'cpufreq_0' in t or 'cpufreq-0' in t:
            affected = [0, 1, 2, 3]
        elif 'cpufreq_1' in t or 'cpufreq-1' in t:
            affected = [4, 5]
        elif 'cpufreq_2' in t or 'cpufreq-2' in t:
            affected = [6, 7]
        else:
            affected = list(range(8))

        for c in affected:
            if c in results:
                results[c]['throttled_by_cooling'] = True
                results[c]['throttled'] = True

    return results

# =============================================================================
# Plot
# =============================================================================
CPU_CLUSTER_CFG = {
    'A55 (0-3)':  {'cores': [0,1,2,3], 'color': '#2196F3'},
    'A76-0 (4-5)':{'cores': [4,5],     'color': '#FF5722'},
    'A76-1 (6-7)':{'cores': [6,7],     'color': '#E91E63'},
}
CPU_CORE_COLORS = [
    '#90CAF9','#64B5F6','#42A5F5','#2196F3',
    '#FF8A65','#FF5722',
    '#F06292','#E91E63',
]

COOLING_PALETTE = [
    '#3F51B5','#009688','#FF5722','#9C27B0',
    '#FF9800','#607D8B','#795548','#F44336',
]

def plot_report(df: pd.DataFrame, save_path: str = None):
    x = df['elapsed_min']
    temp_cols    = [c for c in df.columns if c.startswith('temp_') and c.endswith('_C')]
    cooling_meta = get_cooling_cols(df)
    has_cooling  = len(cooling_meta) > 0
    has_any_flag = 'any_throttle' in df.columns

    # 패널: ① 온도 / ② CPU freq / ③ CPU util / ④ cooling_state
    n_panels = 3 + (1 if has_cooling else 0)
    ratios   = [1.1, 1.0, 0.85, 0.85][:n_panels]

    fig = plt.figure(figsize=(16, 4 * n_panels))
    fig.suptitle('RK3588 CPU Stress Test — Thermal & Throttling Report',
                 fontsize=14, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(n_panels, 1, height_ratios=ratios, hspace=0.40)

    panel = 0

    # ── Panel 0: Temperatures ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[panel]); panel += 1
    for col in temp_cols:
        vals = pd.to_numeric(df[col], errors='coerce')
        label = col.replace('temp_','').replace('_C','').replace('_',' ')
        lw = 2.0 if 'big' in col.lower() else 1.2
        ls = '-' if 'big' in col.lower() else '--'
        ax.plot(x, vals, label=label, linewidth=lw, linestyle=ls)
    ax.axhline(y=80, color='orange', linestyle=':', alpha=0.6, label='80°C warn')
    ax.axhline(y=95, color='red',    linestyle=':', alpha=0.6, label='95°C crit')
    ax.set_ylabel('Temperature (°C)'); ax.set_title('① Thermal Zones')
    ax.legend(loc='upper left', ncol=4, fontsize=7); ax.grid(True, alpha=0.3)

    throttle = detect_throttling(df)

    # ── Panel 1: CPU Frequencies ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[panel], sharex=ax); panel += 1
    for name, cfg in CPU_CLUSTER_CFG.items():
        col = f'cpu{cfg["cores"][0]}_freq_mhz'
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            ax2.plot(x, vals, label=name, linewidth=1.8, color=cfg['color'])
    ax2.set_ylabel('Frequency (MHz)')
    ax2.set_title('② CPU Cluster Frequencies  (drop = freq-based throttle)')
    ax2.legend(loc='lower left', fontsize=9); ax2.grid(True, alpha=0.3)
    for c, info in throttle.items():
        if info['throttled_by_freq'] and c in [0,4,6]:
            ax2.annotate(f'CPU{c}: -{info["drop_pct"]:.0f}%',
                         xy=(x.iloc[-1], info['min_mhz']),
                         fontsize=8, color='red', fontweight='bold')

    # ── Panel 2: CPU Utilization ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[panel], sharex=ax); panel += 1
    for name, cfg in CPU_CLUSTER_CFG.items():
        util_cols = [f'cpu{c}_util_pct' for c in cfg['cores'] if f'cpu{c}_util_pct' in df.columns]
        if not util_cols: continue
        cluster_avg = df[util_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        ax3.fill_between(x, cluster_avg, alpha=0.18, color=cfg['color'])
        ax3.plot(x, cluster_avg, label=f'{name} avg', linewidth=2.0, color=cfg['color'])
        for col in util_cols:
            cid = int(col.replace('cpu','').replace('_util_pct',''))
            ax3.plot(x, pd.to_numeric(df[col], errors='coerce'),
                     linewidth=0.6, linestyle='--', alpha=0.35, color=CPU_CORE_COLORS[cid])
    ax3.set_ylabel('CPU Util (%)'); ax3.set_ylim(-5, 105)
    ax3.set_title('③ CPU Core Utilization')
    ax3.legend(loc='lower left', ncol=3, fontsize=9); ax3.grid(True, alpha=0.3)

    # ── Panel 3: Cooling Device cur_state ────────────────────────────────────
    if has_cooling:
        ax4 = fig.add_subplot(gs[panel], sharex=ax); panel += 1

        global_max = 1
        for meta in cooling_meta:
            vals = pd.to_numeric(df[meta['col']], errors='coerce').fillna(0)
            m = int(vals.max()) if not vals.empty else 0
            global_max = max(global_max, m)

        for i, meta in enumerate(cooling_meta):
            vals = pd.to_numeric(df[meta['col']], errors='coerce').fillna(0)
            label = meta['type'].replace('_', '-')
            color = COOLING_PALETTE[i % len(COOLING_PALETTE)]
            ax4.step(x, vals, label=label, linewidth=1.6,
                     color=color, where='post', alpha=0.85)
            ax4.fill_between(x, vals, step='post', alpha=0.12, color=color)

        if has_any_flag:
            any_t = pd.to_numeric(df['any_throttle'], errors='coerce').fillna(0)
            ax4.fill_between(x, 0, global_max * any_t,
                             alpha=0.08, color='red', label='any_throttle region')

        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax4.set_ylabel('cur_state (0=normal)')
        ax4.set_ylim(-0.3, global_max + 0.5)
        ax4.set_title('④ Cooling Device cur_state  ← Throttling Ground Truth')
        ax4.legend(loc='upper left', ncol=3, fontsize=8)
        ax4.grid(True, alpha=0.3)

        total = len(df)
        for i, meta in enumerate(cooling_meta):
            vals = pd.to_numeric(df[meta['col']], errors='coerce').fillna(0)
            throttle_pct = (vals > 0).sum() / total * 100
            if throttle_pct > 0:
                ax4.text(0.01, 0.92 - i * 0.12,
                         f'{meta["type"].replace("_","-")}: throttled {throttle_pct:.1f}% of time',
                         transform=ax4.transAxes, fontsize=8,
                         color=COOLING_PALETTE[i % len(COOLING_PALETTE)],
                         fontweight='bold')

    # ── 하단 Summary Text ────────────────────────────────────────────────────
    summary = []
    for name, cfg in CPU_CLUSTER_CFG.items():
        info = throttle.get(cfg['cores'][0], {})
        if info.get('throttled'):
            causes = []
            if info.get('throttled_by_freq'):    causes.append('freq↓')
            if info.get('throttled_by_cooling'): causes.append('cooling')
            summary.append(f'{name}: ⚠THROTTLE({"+".join(causes)})')
        elif info:
            summary.append(f'{name}: ✓ {info["init_mhz"]:.0f}MHz')
    if has_cooling and has_any_flag:
        any_t = pd.to_numeric(df['any_throttle'], errors='coerce').fillna(0)
        pct = (any_t > 0).sum() / len(df) * 100
        summary.append(f'Total throttle time: {pct:.1f}%')

    fig.text(0.5, 0.003, ' | '.join(summary), ha='center', fontsize=9,
             style='italic', color='#444')

    plt.tight_layout(rect=[0, 0.015, 1, 0.985])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

# =============================================================================
# CLI Summary
# =============================================================================
def print_summary(df: pd.DataFrame):
    print("\n" + "="*70)
    print("  CPU THROTTLING & UTILIZATION ANALYSIS SUMMARY")
    print("="*70)

    throttle = detect_throttling(df)

    # --- Cooling Device ---
    cooling_meta = get_cooling_cols(df)
    if cooling_meta:
        print("\n  ★ [Cooling Device cur_state — Ground Truth]")
        print(f"  {'Device':<35} {'Throttle%':>10}  {'Max state':>10}  {'Verdict'}")
        print(f"  {'-'*35} {'-'*10}  {'-'*10}  {'-'*15}")
        for meta in cooling_meta:
            vals = pd.to_numeric(df[meta['col']], errors='coerce').fillna(0)
            pct  = (vals > 0).sum() / len(df) * 100
            mx   = int(vals.max())
            verdict = "⚠ THROTTLED" if pct > 0 else "✓ clean"
            print(f"  {meta['type']:<35} {pct:>9.1f}%  {mx:>10}  {verdict}")
        if 'any_throttle' in df.columns:
            any_t = pd.to_numeric(df['any_throttle'], errors='coerce').fillna(0)
            print(f"\n  → Any-throttle duration: {(any_t>0).sum()/len(df)*100:.1f}% of test time")

    # --- CPU Frequency ---
    print("\n  [CPU Frequency — freq drop detection]")
    for c in sorted(throttle):
        info = throttle[c]
        causes = []
        if info['throttled_by_freq']:    causes.append('freq_drop')
        if info['throttled_by_cooling']: causes.append('cooling_state')
        status = f"⚠ ({'+'.join(causes)})" if info['throttled'] else "✓ OK"
        print(f"    CPU{c}: {info['init_mhz']:.0f}→{info['min_mhz']:.0f} MHz "
              f"(drop {info['drop_pct']:+.1f}%)  {status}")

    # --- CPU Utilization ---
    print("\n  [CPU Utilization]")
    for name, cfg in CPU_CLUSTER_CFG.items():
        util_cols = [f'cpu{c}_util_pct' for c in cfg['cores'] if f'cpu{c}_util_pct' in df.columns]
        if not util_cols: continue
        vals = df[util_cols].apply(pd.to_numeric, errors='coerce')
        print(f"    {name:15s}: avg={vals.mean().mean():.1f}%  "
              f"min={vals.min().min():.0f}%  max={vals.max().max():.0f}%")

    # --- Temperatures ---
    print("\n  [Temperatures]")
    temp_cols = [c for c in df.columns if c.startswith('temp_') and c.endswith('_C')]
    for col in temp_cols:
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if vals.empty: continue
        label = col.replace('temp_','').replace('_C','')
        print(f"    {label:25s}: {vals.min():.1f}→{vals.max():.1f}°C  "
              f"avg={vals.mean():.1f}°C  Δ={vals.max()-vals.min():.1f}°C")

    duration = df['elapsed_sec'].max()
    print(f"\n  Duration: {duration/60:.1f} min  |  Samples: {len(df)}")
    print("="*70)

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RK3588 CPU Thermal Log Analyzer')
    parser.add_argument('csv',  help='Path to thermal_log CSV')
    parser.add_argument('--save', '-s', help='Save plot to PNG file')
    args = parser.parse_args()

    df = load_csv(args.csv)
    print_summary(df)
    plot_report(df, save_path=args.save)