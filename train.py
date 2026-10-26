"""
train.py — Fast Non-Blocking Master Training Script
==================================================
Run from project root:
    python -u train.py
"""

import os
import sys
import time
import numpy as np

sys.path.append("src")

from workload_generator import WorkloadGenerator
from lstm_predictor import LSTMPredictor
from rl_agent import (
    train_ppo_agent,
    evaluate_agent,
    evaluate_rule_based,
    save_agent,
)
from multi_agent_scaler import (
    MultiAgentScaler,
    evaluate_single_agent,
    evaluate_rule_based as mas_rule_based,
    print_final_summary,
)
from anomaly_detector import AnomalyDetector


def print_section(title):
    print("\n" + "=" * 60, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 60, flush=True)


def main():

    start_time = time.time()

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print_section("STEP 1 — Generating Workloads")

    gen = WorkloadGenerator(total_steps=1000, seed=42)
    normal_workload, spike_workload, wave_workload = gen.save_all()

    combined_workload = np.concatenate([
        normal_workload,
        spike_workload,
        wave_workload
    ])

    print("✅ Skipped workload plotting during training", flush=True)
    print(f"✅ Workloads ready — {len(spike_workload)} steps each", flush=True)
    print(f"✅ Combined workload ready — {len(combined_workload)} steps", flush=True)

    print_section("STEP 2 — Training LSTM Predictor")

    lstm = LSTMPredictor(sequence_length=10)

    # LSTM learns all traffic patterns
    losses = lstm.train(combined_workload, epochs=20)
    lstm.save("models/lstm_model.pth")

    print("✅ LSTM training complete — model saved", flush=True)
    print("✅ Skipped LSTM prediction plots during training", flush=True)

    print_section("STEP 3 — Training Anomaly Detector")

    detector = AnomalyDetector(contamination=0.05)
    detector.train(normal_workload)

    print("✅ Anomaly detector trained", flush=True)
    print("✅ Skipped anomaly plot during training", flush=True)

    print_section("STEP 4 — Training Single PPO Agent")
    print("  Training Single RL on spike workload...", flush=True)

    # RL trains on spike workload to learn scaling behavior
    model, callback = train_ppo_agent(
        spike_workload,
        lstm,
        total_timesteps=30000,
        name="spike_agent"
    )

    save_agent(model, "ppo_spike_agent")

    print("✅ Single PPO agent trained and saved", flush=True)

    print_section("STEP 5 — Training Multi-Agent System")
    print("  Training Multi-Agent system on spike workload...", flush=True)

    # Multi-Agent trains on spike workload
    mas = MultiAgentScaler(
        workload=spike_workload,
        lstm_predictor=lstm,
        n_agents=3
    )

    mas.train(timesteps=30000)
    mas.save("models")

    print("✅ Multi-agent system trained and saved", flush=True)

    print_section("STEP 6 — Evaluating All Systems")

    rl_results = evaluate_agent(
        model,
        spike_workload,
        lstm,
        agent_name="PPO Agent"
    )

    rule_results_single = evaluate_rule_based(spike_workload)

    multi_results = mas.evaluate()
    single_results = evaluate_single_agent(spike_workload, lstm)
    rule_results_3way = mas_rule_based(spike_workload)

    print_final_summary(
        multi_results,
        single_results,
        rule_results_3way
    )

    elapsed = (time.time() - start_time) / 60

    print_section("TRAINING COMPLETE")
    print(f"  Total time: {elapsed:.1f} minutes", flush=True)
    print("  Models saved to: models/", flush=True)
    print("  Results saved to: results/", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()