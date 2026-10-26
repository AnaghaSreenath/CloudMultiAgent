"""
data_loader.py — Google Cluster Trace 2019 Data Loader
========================================================
UPDATED: Multi-shard support

Training data: Shard 0 + Shard 1 combined
Test data:     Shard 2 only (never used in training)

This fixes the generalization problem found in cross-shard
testing — by training on two different traffic patterns,
the LSTM and RL agents learn more diverse behaviors.

Usage:
    Process training data (Shard 0 + 1):
        python data_loader.py --process

    Process test data (Shard 2):
        python data_loader.py --process-test

    Verify all data:
        python data_loader.py --verify

    Do everything:
        python data_loader.py --all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import json


# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────

DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = DATA_DIR
TEST_DIR = os.path.join(DATA_DIR, 'test')

REQUESTS_PER_CPU_UNIT = 400

# Statistical fallback parameters (Google Cluster Trace 2019)
GOOGLE_TRACE_STATS = {
    'mean_cpu': 0.251,
    'std_cpu': 0.182,
    'diurnal_amplitude': 0.08,
    'burst_probability': 0.03,
    'burst_magnitude': 3.2,
    'autocorrelation': 0.87,
}


# ─────────────────────────────────────────────────────
# CORE: READ ONE SHARD FILE
# ─────────────────────────────────────────────────────

def _read_shard(shard_number, max_records=200000):
    """
    Read one Google Cluster Trace shard file.
    Returns numpy array of CPU utilization values (0-1).
    """
    filename = f"instance_usage-{shard_number:012d}.json.gz"
    path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(path):
        print(f"  ❌ Shard {shard_number} not found: {path}")
        return None

    print(f"  📂 Reading Shard {shard_number} "
          f"({os.path.getsize(path)/1024/1024:.0f} MB)...")

    cpu_values = []
    n_records = 0

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cpu = record.get(
                    'average_usage', {}
                ).get('cpus', None)
                if cpu is not None and cpu >= 0:
                    cpu_values.append(float(cpu))
                    n_records += 1
                if n_records % 50000 == 0 and n_records > 0:
                    print(f"     Parsed {n_records:,} records...")
                if n_records >= max_records:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

    if not cpu_values:
        print(f"  ❌ No CPU data found in Shard {shard_number}")
        return None

    cpu = np.array(cpu_values, dtype=np.float32)
    cpu_max = np.percentile(cpu, 99)
    cpu = np.clip(cpu / (cpu_max + 1e-8), 0.02, 0.98)

    print(f"  ✅ Shard {shard_number}: "
          f"{len(cpu):,} records, "
          f"mean CPU {cpu.mean()*100:.1f}%")

    return cpu


# ─────────────────────────────────────────────────────
# WINDOW EXTRACTION
# ─────────────────────────────────────────────────────

def _extract_normal_window(requests, length=1000):
    """Find most stable low-variance window."""
    series = pd.Series(requests)
    rolling_std = series.rolling(
        window=50
    ).std().fillna(999)

    best_start = 0
    best_std = float('inf')

    for start in range(0, len(requests) - length, 50):
        window_std = rolling_std[
            start:start + length
        ].mean()
        if window_std < best_std:
            best_std = window_std
            best_start = start

    window = requests[
        best_start:best_start + length
    ].copy()
    return np.clip(
        _normalize_to_range(window, 200, 50),
        50, 500
    )


def _extract_spike_window(requests, length=1000):
    """Find window containing biggest traffic burst."""
    series = pd.Series(requests)
    rolling_mean = series.rolling(window=20).mean()
    spike_idx = (series - rolling_mean).idxmax()

    start = max(0, int(spike_idx - length * 0.5))
    end = start + length

    if end > len(requests):
        end = len(requests)
        start = end - length

    window = requests[start:end].copy()
    return np.clip(_normalize_spike(window), 50, 1000)


def _extract_wave_window(requests, length=1000):
    """Find most periodic window."""
    best_start = length
    best_periodicity = 0
    n_windows = (len(requests) - length) // 100

    for i in range(min(n_windows, 20)):
        start = i * 100 + length
        window = requests[start:start + length]
        fft = np.abs(np.fft.rfft(
            window - window.mean()
        ))
        periodicity = fft[1:10].max() / (
            fft[1:].sum() + 1e-8
        )
        if periodicity > best_periodicity:
            best_periodicity = periodicity
            best_start = start

    window = requests[
        best_start:best_start + length
    ].copy()
    return np.clip(
        _normalize_to_range(window, 200, 70),
        50, 500
    )


def _normalize_to_range(data, target_mean=200,
                         target_std=50):
    if data.std() < 1e-6:
        return np.full_like(
            data, target_mean, dtype=np.float32
        )
    normalized = (data - data.mean()) / data.std()
    return (normalized * target_std + target_mean
            ).astype(np.float32)


def _normalize_spike(data):
    series = pd.Series(data)
    rolling_med = series.rolling(
        window=30, center=True
    ).median().fillna(series.median())

    baseline = rolling_med.median()
    spike_val = data.max()

    if spike_val <= baseline + 1e-6:
        return _normalize_to_range(data, 200, 50)

    scale = (800 - 200) / (spike_val - baseline + 1e-8)
    normalized = (data - baseline) * scale + 200
    return normalized.astype(np.float32)


# ─────────────────────────────────────────────────────
# PROCESS TRAINING DATA (Shard 0 + Shard 1 combined)
# ─────────────────────────────────────────────────────

def process_data():
    """
    Combine Shard 0 + Shard 1 into training workloads.

    Why combine?
    - Shard 0 has predictable single-spike patterns
    - Shard 1 has erratic multi-spike patterns
    - Combined = model learns BOTH behaviors
    - Result: better generalization on unseen data

    Output saved to: data/workload_*.npy
    """
    print("=" * 60)
    print("PROCESSING TRAINING DATA")
    print("Combining Shard 0 + Shard 1")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Read both shards
    cpu0 = _read_shard(0)
    cpu1 = _read_shard(1)

    if cpu0 is None and cpu1 is None:
        print("❌ No shards found. Download them first.")
        sys.exit(1)

    # Combine available shards
    available = [c for c in [cpu0, cpu1] if c is not None]
    combined_cpu = np.concatenate(available)
    combined_requests = combined_cpu * REQUESTS_PER_CPU_UNIT

    print(f"\n  Combined dataset:")
    print(f"  Total records: {len(combined_requests):,}")
    print(f"  From {len(available)} shard(s)")

    # Extract three 1000-step windows from combined data
    normal = _extract_normal_window(
        combined_requests, length=1000
    )
    spike = _extract_spike_window(
        combined_requests, length=1000
    )
    wave = _extract_wave_window(
        combined_requests, length=1000
    )

    # Save training workloads
    np.save(
        os.path.join(PROCESSED_DIR, 'workload_normal.npy'),
        normal.astype(np.float32)
    )
    np.save(
        os.path.join(PROCESSED_DIR, 'workload_spike.npy'),
        spike.astype(np.float32)
    )
    np.save(
        os.path.join(PROCESSED_DIR, 'workload_wave.npy'),
        wave.astype(np.float32)
    )

    print(f"\n  ✅ Training workloads saved:")
    print(f"     Normal — mean: {normal.mean():.0f}, "
          f"max: {normal.max():.0f} req/min")
    print(f"     Spike  — mean: {spike.mean():.0f}, "
          f"max: {spike.max():.0f} req/min")
    print(f"     Wave   — mean: {wave.mean():.0f}, "
          f"max: {wave.max():.0f} req/min")
    print(f"\n  Source: Shard 0 + Shard 1 combined")
    print(f"  Shard 2 is reserved for testing only")

    return normal, spike, wave


# ─────────────────────────────────────────────────────
# PROCESS TEST DATA (Shard 2 only — never train on this)
# ─────────────────────────────────────────────────────

def process_test_data():
    """
    Process Shard 2 into test workloads.

    IMPORTANT: Never use this for training.
    This is the final honest evaluation dataset.
    Models have never seen any data from Shard 2.

    Output saved to: data/test/workload_*.npy
    """
    print("=" * 60)
    print("PROCESSING TEST DATA")
    print("Shard 2 — Never seen during training")
    print("=" * 60)

    os.makedirs(TEST_DIR, exist_ok=True)

    cpu2 = _read_shard(2)

    if cpu2 is None:
        print("❌ Shard 2 not found.")
        print("   Download: gsutil cp "
              "gs://clusterdata_2019_a/"
              "instance_usage-000000000002.json.gz "
              "data/raw/")
        sys.exit(1)

    requests = cpu2 * REQUESTS_PER_CPU_UNIT

    # Extract test workloads
    normal = _extract_normal_window(requests, length=1000)
    spike = _extract_spike_window(requests, length=1000)
    wave = _extract_wave_window(requests, length=1000)

    # Save to separate test folder
    np.save(
        os.path.join(TEST_DIR, 'workload_normal.npy'),
        normal.astype(np.float32)
    )
    np.save(
        os.path.join(TEST_DIR, 'workload_spike.npy'),
        spike.astype(np.float32)
    )
    np.save(
        os.path.join(TEST_DIR, 'workload_wave.npy'),
        wave.astype(np.float32)
    )

    print(f"\n  ✅ Test workloads saved to data/test/:")
    print(f"     Normal — mean: {normal.mean():.0f}, "
          f"max: {normal.max():.0f} req/min")
    print(f"     Spike  — mean: {spike.mean():.0f}, "
          f"max: {spike.max():.0f} req/min")
    print(f"     Wave   — mean: {wave.mean():.0f}, "
          f"max: {wave.max():.0f} req/min")
    print(f"\n  ⚠️  These workloads are for testing ONLY")
    print(f"  Do NOT use them in train.py")

    return normal, spike, wave


# ─────────────────────────────────────────────────────
# VERIFY
# ─────────────────────────────────────────────────────

def verify_data():
    """Verify all processed workloads are correct."""
    print("=" * 60)
    print("VERIFYING ALL DATA")
    print("=" * 60)

    all_ok = True

    # Check training data
    print("\n  Training data (data/):")
    for name in ['normal', 'spike', 'wave']:
        path = os.path.join(
            PROCESSED_DIR, f'workload_{name}.npy'
        )
        if os.path.exists(path):
            d = np.load(path)
            issues = []
            if len(d) < 100:
                issues.append("too short")
            if np.isnan(d).any():
                issues.append("contains NaN")
            if d.max() > 1500:
                issues.append(f"max too high: {d.max():.0f}")
            if issues:
                print(f"  ⚠️  {name}: {', '.join(issues)}")
                all_ok = False
            else:
                print(f"  ✅ {name}: {len(d)} steps | "
                      f"mean={d.mean():.0f} | "
                      f"max={d.max():.0f} req/min")
        else:
            print(f"  ❌ {name}: not found")
            all_ok = False

    # Check test data
    print("\n  Test data (data/test/):")
    for name in ['normal', 'spike', 'wave']:
        path = os.path.join(TEST_DIR, f'workload_{name}.npy')
        if os.path.exists(path):
            d = np.load(path)
            print(f"  ✅ {name}: {len(d)} steps | "
                  f"mean={d.mean():.0f} | "
                  f"max={d.max():.0f} req/min")
        else:
            print(f"  ❌ {name}: not found — "
                  f"run --process-test")
            all_ok = False

    # Test CloudEnvironment compatibility
    print("\n  Testing CloudEnvironment compatibility...")
    try:
        sys.path.insert(0, 'src')
        from cloud_env import CloudEnvironment
        spike = np.load(
            os.path.join(PROCESSED_DIR, 'workload_spike.npy')
        )
        env = CloudEnvironment(spike)
        state, _ = env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            state, _, done, _, info = env.step(action)
        print("  ✅ CloudEnvironment compatible")
    except Exception as e:
        print(f"  ❌ CloudEnvironment error: {e}")
        all_ok = False

    # Plot all workloads
    _plot_all_workloads()

    if all_ok:
        print("\n✅ ALL CHECKS PASSED")
        print("  Training data: data/workload_*.npy")
        print("  Test data:     data/test/workload_*.npy")
        print("\n  Next steps:")
        print("  1. python train.py      ← retrain on combined data")
        print("  2. python validate_agents.py ← test on Shard 2")
    else:
        print("\n❌ Some issues found — check above")

    return all_ok


def _plot_all_workloads():
    """Plot training and test workloads side by side."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(
        'Data Overview — Training (Shard 0+1) vs '
        'Test (Shard 2)\nGoogle Cluster Trace 2019',
        fontsize=13, fontweight='bold'
    )

    patterns = ['normal', 'spike', 'wave']
    colors_train = ['steelblue', 'crimson', 'forestgreen']
    colors_test = ['royalblue', 'darkred', 'darkgreen']

    for i, (name, ct, cv) in enumerate(
        zip(patterns, colors_train, colors_test)
    ):
        # Training
        ax = axes[i][0]
        train_path = os.path.join(
            PROCESSED_DIR, f'workload_{name}.npy'
        )
        if os.path.exists(train_path):
            d = np.load(train_path)
            ax.plot(d, color=ct, linewidth=0.8, alpha=0.8)
            ax.axhline(
                y=d.mean(), color='black',
                linestyle='--', linewidth=1,
                alpha=0.4,
                label=f'Mean: {d.mean():.0f}'
            )
            ax.set_title(
                f'Training — {name.capitalize()} '
                f'(Shard 0+1)',
                fontsize=10
            )
        ax.set_ylabel('Requests/min')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Test
        ax = axes[i][1]
        test_path = os.path.join(
            TEST_DIR, f'workload_{name}.npy'
        )
        if os.path.exists(test_path):
            d = np.load(test_path)
            ax.plot(d, color=cv, linewidth=0.8, alpha=0.8)
            ax.axhline(
                y=d.mean(), color='black',
                linestyle='--', linewidth=1,
                alpha=0.4,
                label=f'Mean: {d.mean():.0f}'
            )
            ax.set_title(
                f'Test — {name.capitalize()} (Shard 2)',
                fontsize=10
            )
        else:
            ax.text(
                0.5, 0.5,
                'Not yet processed\nRun --process-test',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=10, color='gray'
            )
            ax.set_title(
                f'Test — {name.capitalize()} (Shard 2)',
                fontsize=10
            )
        ax.set_ylabel('Requests/min')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel('Time Steps')
    axes[-1][1].set_xlabel('Time Steps')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    path = 'results/data_overview.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Data overview saved to {path}")


# ─────────────────────────────────────────────────────
# STATISTICAL FALLBACK
# ─────────────────────────────────────────────────────

def _generate_from_real_stats():
    """Fallback if no shards available."""
    print("  Generating from Google trace statistics...")
    np.random.seed(42)
    n_steps = 5000
    s = GOOGLE_TRACE_STATS
    cpu = np.zeros(n_steps)
    cpu[0] = s['mean_cpu']
    phi = s['autocorrelation']
    sigma = s['std_cpu'] * np.sqrt(1 - phi**2)

    for t in range(1, n_steps):
        diurnal = s['diurnal_amplitude'] * np.sin(
            2 * np.pi * t / 288
        )
        noise = np.random.normal(0, sigma)
        cpu[t] = (
            s['mean_cpu']
            + phi * (cpu[t-1] - s['mean_cpu'])
            + diurnal + noise
        )
        if np.random.random() < s['burst_probability']:
            burst_duration = np.random.randint(3, 15)
            burst_size = np.random.uniform(
                1.5, s['burst_magnitude']
            )
            end = min(t + burst_duration, n_steps)
            cpu[t:end] += burst_size * s['mean_cpu']

    cpu = np.clip(cpu, 0.02, 0.98)
    os.makedirs(RAW_DIR, exist_ok=True)
    df = pd.DataFrame({
        'time': range(n_steps),
        'cpu_util': cpu,
        'source': 'google_trace_2019_stats'
    })
    df.to_csv(
        os.path.join(RAW_DIR, 'machine_usage_processed.csv'),
        index=False
    )
    print(f"  ✅ Generated {n_steps} steps")
    return cpu


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Google Cluster Trace 2019 Data Loader'
    )
    parser.add_argument(
        '--process', action='store_true',
        help='Process Shard 0+1 into training workloads'
    )
    parser.add_argument(
        '--process-test', action='store_true',
        help='Process Shard 2 into test workloads'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify all processed data'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Process training + test + verify'
    )

    args = parser.parse_args()

    if args.all or args.process:
        process_data()

    if args.all or getattr(args, 'process_test', False):
        process_test_data()

    if args.all or args.verify:
        verify_data()

    if not any([
        args.process,
        getattr(args, 'process_test', False),
        args.verify,
        args.all
    ]):
        parser.print_help()
        print("\nQuick start:")
        print("  python data_loader.py --process")
        print("  python data_loader.py --process-test")
        print("  python data_loader.py --verify")


if __name__ == "__main__":
    main()