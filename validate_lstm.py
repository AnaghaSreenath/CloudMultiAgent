"""
validate_lstm.py — LSTM Validation & Overfitting Test
=======================================================
Tests your trained LSTM on data it has NEVER seen.

This script answers three questions:
1. Is the LSTM overfit?        (train vs val loss graph)
2. How accurate are predictions? (MAE, MAPE on test data)
3. Does it generalize?          (Shard 0 vs Shard 1 comparison)

Run from project root:
    python validate_lstm.py

What you'll see:
- Train vs validation loss curves
- Prediction accuracy on Shard 0 (training data)
- Prediction accuracy on Shard 1 (never seen data)
- Side by side comparison graph
- Clear verdict: overfit / underfit / good fit
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')

from lstm_predictor import LSTMPredictor
from data_loader import process_data, _load_raw_trace, PROCESSED_DIR


def load_shard(shard_number):
    """
    Load and process a specific Google Cluster Trace shard.
    Returns spike workload extracted from that shard.
    """
    import gzip
    import json
    import pandas as pd
    from data_loader import (
        RAW_DIR, REQUESTS_PER_CPU_UNIT,
        _extract_spike_window,
        _extract_normal_window,
        _extract_wave_window
    )

    filename = f"instance_usage-{shard_number:012d}.json.gz"
    path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(path):
        print(f"❌ Shard {shard_number} not found at: {path}")
        print(f"   Download it with:")
        print(f"   gsutil cp gs://clusterdata_2019_a/"
              f"instance_usage-{shard_number:012d}.json.gz data/raw/")
        return None

    print(f"📂 Loading Shard {shard_number}: {filename}")
    print(f"   Size: {os.path.getsize(path)/1024/1024:.0f} MB")

    cpu_values = []
    n_records = 0

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                usage = record.get('average_usage', {})
                cpu = usage.get('cpus', None)
                if cpu is not None and cpu >= 0:
                    cpu_values.append(float(cpu))
                    n_records += 1
                if n_records % 50000 == 0 and n_records > 0:
                    print(f"   Parsed {n_records:,} records...")
                if n_records >= 200000:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

    if not cpu_values:
        print(f"❌ No CPU data found in shard {shard_number}")
        return None

    cpu = np.array(cpu_values, dtype=np.float32)
    cpu_max = np.percentile(cpu, 99)
    cpu = np.clip(cpu / cpu_max, 0.02, 0.98)

    requests = cpu * REQUESTS_PER_CPU_UNIT

    spike = _extract_spike_window(requests, length=1000)
    normal = _extract_normal_window(requests, length=1000)
    wave = _extract_wave_window(requests, length=1000)

    print(f"✅ Shard {shard_number} loaded:")
    print(f"   Normal — mean: {normal.mean():.0f} req/min")
    print(f"   Spike  — mean: {spike.mean():.0f}, "
          f"max: {spike.max():.0f} req/min")
    print(f"   Wave   — mean: {wave.mean():.0f} req/min")

    return {
        'normal': normal,
        'spike': spike,
        'wave': wave
    }


def test_overfitting(lstm):
    """
    Test 1: Plot train vs validation loss.
    This is the primary overfitting detection test.
    """
    print("\n" + "=" * 60)
    print("TEST 1 — OVERFITTING DETECTION")
    print("Train Loss vs Validation Loss")
    print("=" * 60)

    if not lstm.train_losses or not lstm.val_losses:
        print("⚠️  No training history in loaded model.")
        print("   Retrain with updated lstm_predictor.py first:")
        print("   python train.py")
        return False

    train_l = lstm.train_losses
    val_l = lstm.val_losses

    final_gap = abs(val_l[-1] - train_l[-1])
    min_val = min(val_l)
    min_val_epoch = val_l.index(min_val)

    print(f"\n   Epochs trained:      {len(train_l)}")
    print(f"   Final train loss:    {train_l[-1]:.6f}")
    print(f"   Final val loss:      {val_l[-1]:.6f}")
    print(f"   Gap:                 {final_gap:.6f}")
    print(f"   Best val loss:       {min_val:.6f} "
          f"at epoch {min_val_epoch+1}")

    if final_gap < 0.003:
        verdict = "✅ GOOD FIT — No overfitting detected"
        detail = ("Training and validation loss are very close. "
                  "Model learned genuine patterns, not memorization.")
    elif final_gap < 0.01:
        verdict = "⚠️  SLIGHT OVERFIT — Acceptable for research"
        detail = ("Small gap between losses. Normal for complex data. "
                  "Model still generalizes well.")
    else:
        verdict = "❌ OVERFIT — Model memorized training data"
        detail = ("Large gap between losses. Model fails on new data. "
                  "Consider: more training data, higher dropout, "
                  "fewer epochs.")

    print(f"\n   {verdict}")
    print(f"   {detail}")

    lstm.plot_train_val_loss(
        save_path='results/lstm_train_val_loss.png'
    )

    return final_gap < 0.01


def test_prediction_accuracy(lstm, workload, label):
    """
    Test 2: How accurate are predictions on this dataset?
    Returns error metrics.
    """
    results = lstm.evaluate_on_data(workload, label=label)

    # Show sample predictions
    print(f"\n   Sample predictions vs actual ({label}):")
    print(f"   {'Step':>5} {'Actual':>10} "
          f"{'Predicted':>10} {'Error':>8} {'%Err':>7}")
    print(f"   {'-'*45}")

    preds = results['predictions']
    acts = results['actuals']

    # Show 10 evenly spaced samples
    indices = np.linspace(0, len(preds)-1, 10, dtype=int)
    for i in indices:
        err = abs(preds[i] - acts[i])
        pct = err / (acts[i] + 1e-8) * 100
        status = "✅" if pct < 10 else "⚠️"
        print(f"   {i:>5} {acts[i]:>10.1f} "
              f"{preds[i]:>10.1f} "
              f"{err:>8.1f} "
              f"{pct:>6.1f}% {status}")

    return results


def test_generalization(lstm, shard0_data, shard1_data):
    """
    Test 3: Does model generalize to unseen Shard 1 data?

    Trains on Shard 0, tests on Shard 1.
    If accuracy is similar — model generalizes. ✅
    If accuracy drops a lot — model is overfit. ❌
    """
    print("\n" + "=" * 60)
    print("TEST 3 — GENERALIZATION TEST")
    print("Shard 0 (trained on) vs Shard 1 (never seen)")
    print("=" * 60)

    # Evaluate on Shard 0 (training data)
    r0 = test_prediction_accuracy(
        lstm, shard0_data['spike'], "Shard 0 (Training Data)"
    )

    # Evaluate on Shard 1 (never seen)
    r1 = test_prediction_accuracy(
        lstm, shard1_data['spike'], "Shard 1 (Unseen Data)"
    )

    # Compare
    mae_diff = abs(r1['mae'] - r0['mae'])
    mape_diff = abs(r1['mape'] - r0['mape'])
    mae_pct_change = (mae_diff / (r0['mae'] + 1e-8)) * 100

    print(f"\n   GENERALIZATION COMPARISON:")
    print(f"   {'Metric':<25} {'Shard 0':>12} "
          f"{'Shard 1':>12} {'Change':>10}")
    print(f"   {'-'*60}")
    print(f"   {'MAE (req/min)':<25} "
          f"{r0['mae']:>12.2f} "
          f"{r1['mae']:>12.2f} "
          f"{mae_pct_change:>+9.1f}%")
    print(f"   {'MAPE (%)':<25} "
          f"{r0['mape']:>12.2f}% "
          f"{r1['mape']:>12.2f}% "
          f"{mape_diff:>+9.1f}%")

    if mae_pct_change < 20:
        verdict = "✅ GENERALIZES WELL — Less than 20% accuracy drop"
        detail = "Model learned real traffic patterns, not Shard 0 specifics."
    elif mae_pct_change < 50:
        verdict = "⚠️  MODERATE DROP — Some overfitting to Shard 0"
        detail = "Consider training on more diverse data."
    else:
        verdict = "❌ POOR GENERALIZATION — Heavily overfit to Shard 0"
        detail = "Model memorized Shard 0. Needs more training data variety."

    print(f"\n   {verdict}")
    print(f"   {detail}")

    # Plot comparison graph
    _plot_generalization(r0, r1, shard0_data, shard1_data)

    return r0, r1


def _plot_generalization(r0, r1, shard0_data, shard1_data):
    """Plot Shard 0 vs Shard 1 predictions side by side."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        'LSTM Generalization Test\n'
        'Shard 0 (Training Data) vs Shard 1 (Never Seen)',
        fontsize=13, fontweight='bold'
    )

    # Shard 0 — full prediction
    ax = axes[0][0]
    ax.plot(r0['actuals'], color='steelblue',
            linewidth=1, label='Actual', alpha=0.8)
    ax.plot(r0['predictions'], color='crimson',
            linewidth=1.5, linestyle='--',
            label='Predicted', alpha=0.9)
    ax.set_title(f"Shard 0 — MAE: {r0['mae']:.1f} req/min "
                 f"| MAPE: {r0['mape']:.1f}%",
                 fontsize=10)
    ax.set_ylabel('Requests/min')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Shard 1 — full prediction
    ax = axes[0][1]
    ax.plot(r1['actuals'], color='steelblue',
            linewidth=1, label='Actual', alpha=0.8)
    ax.plot(r1['predictions'], color='crimson',
            linewidth=1.5, linestyle='--',
            label='Predicted', alpha=0.9)
    ax.set_title(f"Shard 1 — MAE: {r1['mae']:.1f} req/min "
                 f"| MAPE: {r1['mape']:.1f}%",
                 fontsize=10)
    ax.set_ylabel('Requests/min')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Error distribution — Shard 0
    ax = axes[1][0]
    errors0 = np.abs(
        np.array(r0['predictions']) - np.array(r0['actuals'])
    )
    ax.hist(errors0, bins=40, color='steelblue',
            alpha=0.7, edgecolor='white')
    ax.axvline(x=errors0.mean(), color='crimson',
               linestyle='--', linewidth=2,
               label=f'Mean: {errors0.mean():.1f}')
    ax.set_title('Shard 0 — Error Distribution')
    ax.set_xlabel('Absolute Error (req/min)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error distribution — Shard 1
    ax = axes[1][1]
    errors1 = np.abs(
        np.array(r1['predictions']) - np.array(r1['actuals'])
    )
    ax.hist(errors1, bins=40, color='darkorange',
            alpha=0.7, edgecolor='white')
    ax.axvline(x=errors1.mean(), color='crimson',
               linestyle='--', linewidth=2,
               label=f'Mean: {errors1.mean():.1f}')
    ax.set_title('Shard 1 — Error Distribution')
    ax.set_xlabel('Absolute Error (req/min)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = 'results/lstm_generalization_test.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"\n✅ Generalization graph saved to {path}")


def main():

    print("=" * 60)
    print("LSTM VALIDATION SUITE")
    print("Overfitting Detection + Accuracy + Generalization")
    print("=" * 60)

    # ── Load LSTM ──
    print("\n📂 Loading trained LSTM...")
    lstm = LSTMPredictor()
    lstm.load('models/lstm_model.pth')

    # ── Load Shard 0 workloads ──
    print("\n📂 Loading Shard 0 workloads...")
    shard0_data = {
        'normal': np.load('data/workload_normal.npy'),
        'spike': np.load('data/workload_spike.npy'),
        'wave': np.load('data/workload_wave.npy')
    }
    print(f"✅ Shard 0 loaded — "
          f"{len(shard0_data['spike'])} steps each")

    # ── TEST 1 — Overfitting detection ──
    has_history = test_overfitting(lstm)

    if not has_history:
        print("\n⚠️  Skipping overfitting test.")
        print("   To enable: retrain with updated lstm_predictor.py")
        print("   Run: python train.py")

    # ── TEST 2 — Prediction accuracy on Shard 0 ──
    print("\n" + "=" * 60)
    print("TEST 2 — PREDICTION ACCURACY ON SHARD 0")
    print("(Data model was trained on)")
    print("=" * 60)
    test_prediction_accuracy(
        lstm, shard0_data['spike'], "Shard 0 Spike"
    )
    test_prediction_accuracy(
        lstm, shard0_data['normal'], "Shard 0 Normal"
    )
    test_prediction_accuracy(
        lstm, shard0_data['wave'], "Shard 0 Wave"
    )

    # ── TEST 3 — Generalization on Shard 1 ──
    print("\n📂 Loading Shard 1 (unseen data)...")
    shard1_data = load_shard(1)

    if shard1_data is None:
        print("\n⚠️  Shard 1 not available.")
        print("   Download it first:")
        print("   gsutil cp gs://clusterdata_2019_a/"
              "instance_usage-000000000001.json.gz data/raw/")
        print("\n   Skipping generalization test.")
    else:
        test_generalization(lstm, shard0_data, shard1_data)

    # ── FINAL SUMMARY ──
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print("\nGraphs saved to results/:")
    print("  lstm_train_val_loss.png      ← overfitting detection")
    print("  lstm_generalization_test.png ← shard comparison")
    print("\nNext step: python validate_agents.py")
    print("=" * 60)


if __name__ == "__main__":
    main()