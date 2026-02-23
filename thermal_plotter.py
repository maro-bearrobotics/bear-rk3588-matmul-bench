#!/usr/bin/env python3
"""
RK3588 Thermal & Throttling Analysis Plotter
Usage: python3 plot_thermal.py /tmp/thermal_log_*.csv [--save output.png]
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment='#')
    df['elapsed_min'] = df['elapsed_sec'] / 60.0
    return df

def detect_throttling(df: pd.DataFrame) -> dict:
    results = {}
    for c in range(8):
        col = f'cpu{c}_freq_mhz'
        if col not in df.columns:
            continue
        freq = pd.to_numeric(df[col], errors='coerce').dropna()
        if freq.empty:
            continue
        max_f = freq.max()
        min_f = freq.min()
        init_f = freq.head(10).mean()
        drop_pct = (1 - min_f / init_f) * 100 if init_f > 0 else 0
        results[c] = {
            'max_mhz': max_f, 'min_mhz': min_f,
            'init_mhz': init_f, 'drop_pct': drop_pct,
            'throttled': drop_pct > 5
        }
    return results

def plot_report(df: pd.DataFrame, save_path: str = None):
    x = df['elapsed_min']

    fig = plt.figure(figsize=(16, 18))
    fig.suptitle('RK3588 CPU + NPU Stress Test — Thermal & Throttling Report',
                 fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 0.8, 0.8, 0.8], hspace=0.38)

    # --- Panel 1: Temperatures ---
    ax1 = fig.add_subplot(gs[0])
    temp_cols = [c for c in df.columns if c.startswith('temp_') and c.endswith('_C')]

    for col in temp_cols:
        vals = pd.to_numeric(df[col], errors='coerce')
        label = col.replace('temp_', '').replace('_C', '').replace('_', ' ')
        lw = 2.0 if 'npu' in col.lower() else 1.2
        ls = '-' if 'npu' in col.lower() or 'big' in col.lower() else '--'
        ax1.plot(x, vals, label=label, linewidth=lw, linestyle=ls)

    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Thermal Zones')
    ax1.legend(loc='upper left', ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=80, color='red', linestyle=':', alpha=0.5)
    ax1.axhline(y=95, color='darkred', linestyle=':', alpha=0.5)

    # --- Panel 2: CPU Frequencies ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    cluster_colors = {
        'A55 (0-3)': '#2196F3',
        'A76-0 (4-5)': '#FF5722',
        'A76-1 (6-7)': '#E91E63',
    }
    cpu_clusters = {
        'A55 (0-3)': [0, 1, 2, 3],
        'A76-0 (4-5)': [4, 5],
        'A76-1 (6-7)': [6, 7],
    }
    for name, cores in cpu_clusters.items():
        col = f'cpu{cores[0]}_freq_mhz'
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            ax2.plot(x, vals, label=name, linewidth=1.8, color=cluster_colors[name])

    ax2.set_ylabel('Frequency (MHz)')
    ax2.set_title('CPU Cluster Frequencies (drop = throttling)')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    throttle = detect_throttling(df)
    for c, info in throttle.items():
        if info['throttled']:
            ax2.annotate(f'CPU{c}: -{info["drop_pct"]:.0f}%',
                        xy=(x.iloc[-1], info['min_mhz']),
                        fontsize=8, color='red', fontweight='bold')

    # --- Panel 3: CPU Core Utilization ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    cpu_colors_per_core = [
        '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3',  # A55: light→dark blue
        '#FF8A65', '#FF5722',                          # A76-0: orange
        '#F06292', '#E91E63',                          # A76-1: pink
    ]
    # Plot per-cluster averages as filled areas + individual cores as thin lines
    for name, cores in cpu_clusters.items():
        util_cols = [f'cpu{c}_util_pct' for c in cores if f'cpu{c}_util_pct' in df.columns]
        if not util_cols:
            continue
        cluster_avg = df[util_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        color = cluster_colors[name]
        ax3.fill_between(x, cluster_avg, alpha=0.2, color=color)
        ax3.plot(x, cluster_avg, label=f'{name} avg', linewidth=2.0, color=color)
        # Individual cores as thin dashed lines
        for col in util_cols:
            core_id = col.replace('cpu', '').replace('_util_pct', '')
            vals = pd.to_numeric(df[col], errors='coerce')
            ax3.plot(x, vals, linewidth=0.6, linestyle='--', alpha=0.4,
                     color=cpu_colors_per_core[int(core_id)])

    ax3.set_ylabel('CPU Util (%)')
    ax3.set_ylim(-5, 105)
    ax3.set_title('CPU Core Utilization (by cluster)')
    ax3.legend(loc='lower left', ncol=3, fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=100, color='gray', linestyle=':', alpha=0.3)
    ax3.axhline(y=50, color='gray', linestyle=':', alpha=0.2)

    # --- Panel 4: NPU Core Utilization ---
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    npu_colors = ['#4CAF50', '#FF9800', '#9C27B0']
    for i in range(3):
        col = f'npu_core{i}_pct'
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            ax4.fill_between(x, vals, alpha=0.3, color=npu_colors[i])
            ax4.plot(x, vals, label=f'Core{i}', linewidth=1.2, color=npu_colors[i])

    ax4.set_ylabel('NPU Load (%)')
    ax4.set_ylim(-5, 105)
    ax4.set_title('NPU Core Utilization')
    ax4.legend(loc='lower left', ncol=3, fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=100, color='green', linestyle=':', alpha=0.4)

    # --- Panel 5: NPU Temp vs Load ---
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    npu_temp_col = [c for c in temp_cols if 'npu' in c.lower()]
    if npu_temp_col:
        t = pd.to_numeric(df[npu_temp_col[0]], errors='coerce')
        ax5_twin = ax5.twinx()
        ax5.plot(x, t, color='#F44336', linewidth=2, label='NPU Temp')
        ax5.set_ylabel('NPU Temp (°C)', color='#F44336')

        npu_avg = pd.to_numeric(df.get('npu_core0_pct', 0), errors='coerce').fillna(0)
        for i in [1, 2]:
            col = f'npu_core{i}_pct'
            if col in df.columns:
                npu_avg = npu_avg + pd.to_numeric(df[col], errors='coerce').fillna(0)
        npu_avg = npu_avg / 3.0
        ax5_twin.plot(x, npu_avg, color='#4CAF50', linewidth=1.5, alpha=0.7, label='Avg NPU Load')
        ax5_twin.set_ylabel('Avg NPU Load (%)', color='#4CAF50')
        ax5_twin.set_ylim(-5, 105)

    ax5.set_xlabel('Elapsed Time (minutes)')
    ax5.set_title('NPU Temperature vs Load Correlation')
    ax5.grid(True, alpha=0.3)

    # --- Summary ---
    summary = []
    if npu_temp_col:
        npu_t = pd.to_numeric(df[npu_temp_col[0]], errors='coerce')
        summary.append(f'NPU Temp: {npu_t.min():.0f}→{npu_t.max():.0f}°C')
    for name, cores in cpu_clusters.items():
        info = throttle.get(cores[0], {})
        if info.get('throttled'):
            summary.append(f'{name}: THROTTLED (-{info["drop_pct"]:.0f}%)')
        elif info:
            summary.append(f'{name}: stable @ {info["init_mhz"]:.0f}MHz')
    # CPU util summary
    for name, cores in cpu_clusters.items():
        util_cols = [f'cpu{c}_util_pct' for c in cores if f'cpu{c}_util_pct' in df.columns]
        if util_cols:
            avg = df[util_cols].apply(pd.to_numeric, errors='coerce').mean().mean()
            summary.append(f'{name} util: {avg:.0f}%')

    fig.text(0.5, 0.005, ' | '.join(summary), ha='center', fontsize=9,
             style='italic', color='#555')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

def print_summary(df: pd.DataFrame):
    print("\n" + "="*65)
    print("  THROTTLING & UTILIZATION ANALYSIS SUMMARY")
    print("="*65)

    throttle = detect_throttling(df)
    print("\n  [CPU Frequency]")
    for c in sorted(throttle):
        info = throttle[c]
        status = "⚠ THROTTLED" if info['throttled'] else "✓ OK"
        print(f"    CPU{c}: {info['init_mhz']:.0f} → {info['min_mhz']:.0f} MHz "
              f"({-info['drop_pct']:+.1f}%) {status}")

    print("\n  [CPU Utilization]")
    cpu_clusters = {'A55 (0-3)': [0,1,2,3], 'A76-0 (4-5)': [4,5], 'A76-1 (6-7)': [6,7]}
    for name, cores in cpu_clusters.items():
        util_cols = [f'cpu{c}_util_pct' for c in cores if f'cpu{c}_util_pct' in df.columns]
        if util_cols:
            vals = df[util_cols].apply(pd.to_numeric, errors='coerce')
            avg = vals.mean().mean()
            mn = vals.min().min()
            mx = vals.max().max()
            print(f"    {name:15s}: avg={avg:.1f}%  min={mn:.0f}%  max={mx:.0f}%")

    print("\n  [NPU Utilization]")
    for i in range(3):
        col = f'npu_core{i}_pct'
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            print(f"    Core{i}:          avg={vals.mean():.1f}%  "
                  f"min={vals.min():.0f}%  max={vals.max():.0f}%")

    print("\n  [Temperatures]")
    temp_cols = [c for c in df.columns if c.startswith('temp_') and c.endswith('_C')]
    for col in temp_cols:
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if vals.empty:
            continue
        label = col.replace('temp_', '').replace('_C', '')
        print(f"    {label:20s}: min={vals.min():.1f}  max={vals.max():.1f}  "
              f"avg={vals.mean():.1f}°C  Δ={vals.max()-vals.min():.1f}°C")

    duration = df['elapsed_sec'].max()
    print(f"\n  Test duration: {duration/60:.1f} min ({duration:.0f} sec)")
    print(f"  Samples: {len(df)}")
    print("="*65)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RK3588 Thermal Log Analyzer')
    parser.add_argument('csv', help='Path to thermal_log CSV')
    parser.add_argument('--save', '-s', help='Save plot to file (e.g. report.png)')
    args = parser.parse_args()

    df = load_csv(args.csv)
    print_summary(df)
    plot_report(df, save_path=args.save)