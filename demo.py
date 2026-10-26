"""
demo.py — Quick Terminal Demo
================================
Loads all trained models and runs a fast evaluation.
Does NOT retrain anything — just shows results.

Run from project root:
    python demo.py
"""

import sys
import os
import numpy as np

sys.path.append('src')

from workload_generator import WorkloadGenerator
from lstm_predictor import LSTMPredictor
from anomaly_detector import AnomalyDetector
from multi_agent_scaler import (
    MultiAgentScaler,
    evaluate_single_agent,
    evaluate_rule_based,
    plot_three_way_comparison,
    print_final_summary
)
from stable_baselines3 import PPO


def check_models():
    """Make sure all trained models exist before running."""
    required = [
        'models/lstm_model.pth',
        'models/ppo_spike_agent.zip',
        'models/multi_agent_1.zip',
        'models/multi_agent_2.zip',
        'models/multi_agent_3.zip',
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("❌ Missing trained models:")
        for f in missing:
            print(f"   {f}")
        print("\nRun `python train.py` first to train all models.")
        sys.exit(1)
    print("✅ All models found")


def print_section(title):
    print("\n" + "─" * 55)
    print(f"  {title}")
    print("─" * 55)


def main():

    print("\n" + "=" * 55)
    print("  🌩️  Cloud Autoscaler — Demo")
    print("  Predictive Multi-Agent RL vs Rule-Based")
    print("=" * 55)

    # ── Check models exist ──
    check_models()

    # ── Load data ──
    print_section("Loading Data & Models")

    gen = WorkloadGenerator(total_steps=1000, seed=42)
    spike_workload = gen.generate_spike()
    normal_workload = gen.generate_normal()

    lstm = LSTMPredictor()
    lstm.load('models/lstm_model.pth')

    detector = AnomalyDetector(contamination=0.05)
    detector.train(normal_workload)

    mas = MultiAgentScaler(
        workload=spike_workload,
        lstm_predictor=lstm,
        n_agents=3
    )
    mas.load('models')

    print("✅ All models loaded")

    # ── LSTM prediction sample ──
    print_section("LSTM Prediction Sample")
    print("  Predicting next traffic value from recent history:")
    print()

    test_sequences = [
        spike_workload[490:500],   # just before spike
        spike_workload[500:510],   # during spike
        spike_workload[580:590],   # after spike
    ]
    labels = [
        "Before spike  (steps 490–500)",
        "During spike  (steps 500–510)",
        "After spike   (steps 580–590)",
    ]

    for label, seq in zip(labels, test_sequences):
        actual_next = spike_workload[
            list(spike_workload).index(seq[-1]) + 1
            if list(spike_workload).index(seq[-1]) + 1
            < len(spike_workload) else -1
        ] if False else "—"
        pred = lstm.predict_next(seq)
        print(f"  {label}")
        print(f"    History: [{', '.join(f'{v:.0f}' for v in seq)}]")
        print(f"    Predicted next: {pred:.1f} req/min")
        print()

    # ── Anomaly detection sample ──
    print_section("Anomaly Detection")

    anomalies = detector.scan_workload(spike_workload)
    n_anomalies = sum(anomalies)
    pct = n_anomalies / len(spike_workload) * 100

    print(f"  Scanning {len(spike_workload)} time steps...")
    print(f"  Anomalies detected: {n_anomalies} ({pct:.1f}%)")
    print(f"  Normal points:      {len(spike_workload) - n_anomalies}")

    # Show where anomalies cluster
    anom_indices = [i for i, a in enumerate(anomalies) if a]
    if anom_indices:
        print(f"  First anomaly at step: {min(anom_indices)}")
        print(f"  Last anomaly at step:  {max(anom_indices)}")
        mid = len(spike_workload) // 2
        spike_anomalies = sum(
            1 for i in anom_indices
            if mid - 50 <= i <= mid + 130
        )
        print(f"  Anomalies in spike zone (steps 450–630): "
              f"{spike_anomalies}")

    # ── Three-way evaluation ──
    print_section("Running Three-Way Evaluation")
    print("  Rule-Based  →  Single RL Agent  →  Multi-Agent RL")
    print()

    multi_results = mas.evaluate()
    single_results = evaluate_single_agent(spike_workload, lstm)
    rule_results = evaluate_rule_based(spike_workload)

    # ── Final summary table ──
    print_final_summary(multi_results, single_results, rule_results)

    # ── Generate comparison plot ──
    print_section("Generating Comparison Plot")
    plot_three_way_comparison(
        multi_results,
        single_results,
        rule_results,
        spike_workload
    )

    # ── Improvement summary ──
    print_section("Key Improvements vs Rule-Based")

    rm = rule_results['metrics']
    sm = single_results['metrics']
    mm = multi_results['metrics']

    def pct_change(new, old):
        if old == 0:
            return "N/A"
        return f"{((new - old) / old) * 100:+.1f}%"

    print(f"  {'Metric':<25} {'Single RL':>12} {'Multi-Agent':>12}")
    print(f"  {'-'*50}")
    print(f"  {'SLA Violations':<25} "
          f"{pct_change(sm['total_sla_violations'], rm['total_sla_violations']):>12} "
          f"{pct_change(mm['total_sla_violations'], rm['total_sla_violations']):>12}")
    print(f"  {'Cost':<25} "
          f"{pct_change(sm['total_cost'], rm['total_cost']):>12} "
          f"{pct_change(mm['total_cost'], rm['total_cost']):>12}")
    print(f"  {'Scaling Actions':<25} "
          f"{pct_change(sm['scaling_actions'], rm['scaling_actions']):>12} "
          f"{pct_change(mm['scaling_actions'], rm['scaling_actions']):>12}")

    print("\n  ✅ Demo complete!")
    print("  Launch dashboard:  cd dashboard && streamlit run dashboard.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()