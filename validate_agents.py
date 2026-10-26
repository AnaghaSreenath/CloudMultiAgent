"""
validate_agents.py — RL Agent Generalization Test
================================================
Run from project root:
    python validate_agents.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("src")

from stable_baselines3 import PPO
from cloud_env import CloudEnvironment
from lstm_predictor import LSTMPredictor
from multi_agent_scaler import MultiAgentScaler


def run_rule_based(workload):
    env = CloudEnvironment(workload)
    state, _ = env.reset()

    vm_hist, resp_hist, sla_hist = [], [], []
    done = False

    while not done:
        action = env.rule_based_action(state)
        state, _, done, _, info = env.step(action)

        vm_hist.append(info["vms"])
        resp_hist.append(info["response_time"])
        sla_hist.append(info["sla_violated"])

    return {
        "vm_history": vm_hist,
        "response_history": resp_hist,
        "sla_history": sla_hist,
        "metrics": env.get_metrics()
    }


def run_single_agent(workload, lstm):
    model = PPO.load("models/ppo_spike_agent")
    env = CloudEnvironment(workload, lstm)
    state, _ = env.reset()

    vm_hist, resp_hist, sla_hist = [], [], []
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, _, done, _, info = env.step(action)

        vm_hist.append(info["vms"])
        resp_hist.append(info["response_time"])
        sla_hist.append(info["sla_violated"])

    return {
        "vm_history": vm_hist,
        "response_history": resp_hist,
        "sla_history": sla_hist,
        "metrics": env.get_metrics()
    }


def run_multi_agent(workload, lstm):
    """
    Corrected version:
    Uses MultiAgentScaler.evaluate() directly.
    This avoids old manual segment-based validation.
    """
    mas = MultiAgentScaler(
        workload=workload,
        lstm_predictor=lstm,
        n_agents=3
    )

    mas.load("models")

    result = mas.evaluate()

    result["sla_history"] = [
        rt > 2.0 for rt in result["response_history"]
    ]

    return result


def evaluate_all_systems(workload, lstm, label):
    print(f"\n  Running Rule-Based on {label}...")
    rule = run_rule_based(workload)

    print(f"  Running Single RL on {label}...")
    single = run_single_agent(workload, lstm)

    print(f"  Running Multi-Agent on {label}...")
    multi = run_multi_agent(workload, lstm)

    print(f"\n  Results on {label}:")
    print(f"  {'System':<15} {'SLA Violations':>15} {'Cost':>10} {'Scaling':>10}")
    print(f"  {'-' * 52}")

    for name, result in [
        ("Rule-Based", rule),
        ("Single RL", single),
        ("Multi-Agent", multi)
    ]:
        m = result["metrics"]
        print(
            f"  {name:<15} "
            f"{m['total_sla_violations']:>15} "
            f"{m['total_cost']:>10.1f} "
            f"{m['scaling_actions']:>10}"
        )

    return rule, single, multi


def plot_cross_shard_comparison(
    s0_rule, s0_single, s0_multi,
    s2_rule, s2_single, s2_multi,
    shard0_workload, shard2_workload
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    fig.suptitle(
        "Agent Generalization Test\n"
        "Training Data vs Test Data (Shard 2 — Never Seen)",
        fontsize=13,
        fontweight="bold"
    )

    def plot_row(axes_row, rule, single, multi, title):
        ax = axes_row[0]
        ax.plot(multi["vm_history"], label="Multi-Agent RL")
        ax.plot(single["vm_history"], linestyle="--", label="Single RL")
        ax.plot(rule["vm_history"], linestyle=":", label="Rule-Based")
        ax.set_title(f"{title} — VM Count")
        ax.set_ylabel("VMs")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes_row[1]
        ax.plot(multi["response_history"], label="Multi-Agent RL")
        ax.plot(single["response_history"], linestyle="--", label="Single RL")
        ax.plot(rule["response_history"], linestyle=":", label="Rule-Based")
        ax.axhline(y=2.0, linestyle=":", linewidth=2, label="SLA Threshold")
        ax.set_title(f"{title} — Response Time")
        ax.set_ylabel("Seconds")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes_row[2]
        systems = ["Rule\nBased", "Single\nRL", "Multi\nAgent"]
        violations = [
            rule["metrics"]["total_sla_violations"],
            single["metrics"]["total_sla_violations"],
            multi["metrics"]["total_sla_violations"]
        ]

        bars = ax.bar(systems, violations, alpha=0.8)

        for bar, val in zip(bars, violations):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(val),
                ha="center",
                fontweight="bold"
            )

        ax.set_title(f"{title} — SLA Violations")
        ax.set_ylabel("Violations")
        ax.grid(True, alpha=0.3, axis="y")

    plot_row(
        axes[0],
        s0_rule,
        s0_single,
        s0_multi,
        "Training Data"
    )

    plot_row(
        axes[1],
        s2_rule,
        s2_single,
        s2_multi,
        "Test Data Shard 2"
    )

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    path = "results/agent_generalization_test.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")

    # Important: do not block terminal
    plt.close(fig)

    print(f"Generalization graph saved to {path}")


def print_generalization_verdict(
    s0_rule, s0_single, s0_multi,
    s2_rule, s2_single, s2_multi
):
    print("\n" + "=" * 60)
    print("GENERALIZATION VERDICT")
    print("=" * 60)

    s0_r = s0_rule["metrics"]["total_sla_violations"]
    s0_m = s0_multi["metrics"]["total_sla_violations"]

    s2_r = s2_rule["metrics"]["total_sla_violations"]
    s2_m = s2_multi["metrics"]["total_sla_violations"]

    print(f"\nTraining Rule-Based SLA:     {s0_r}")
    print(f"Training Multi-Agent SLA:    {s0_m}")
    print(f"Shard 2 Rule-Based SLA:      {s2_r}")
    print(f"Shard 2 Multi-Agent SLA:     {s2_m}")

    if s2_m <= s2_r:
        print("\nVerdict: GOOD GENERALIZATION")
        print("Multi-Agent is matching or beating Rule-Based on unseen data.")
    elif s2_m <= s2_r + 3:
        print("\nVerdict: PARTIAL GENERALIZATION")
        print("Multi-Agent is close, but still needs tuning.")
    else:
        print("\nVerdict: WEAK GENERALIZATION")
        print("Multi-Agent is still not safe enough on unseen traffic.")

    print("=" * 60)


def main():
    print("=" * 60)
    print("AGENT GENERALIZATION TEST")
    print("Training vs Shard 2 Test")
    print("=" * 60)

    print("\nLoading LSTM...")
    lstm = LSTMPredictor()
    lstm.load("models/lstm_model.pth")

    print("\nLoading training workloads...")
    shard0_spike = np.load("data/workload_spike.npy")

    print("\nLoading Shard 2 test workloads...")
    test_spike = "data/test/workload_spike.npy"

    if not os.path.exists(test_spike):
        print("Shard 2 test data not found.")
        print("Run first: python data_loader.py --process-test")
        sys.exit(1)

    shard2_spike = np.load(test_spike)

    print("\n" + "=" * 60)
    print("EVALUATING ON TRAINING DATA")
    print("=" * 60)

    s0_rule, s0_single, s0_multi = evaluate_all_systems(
        shard0_spike,
        lstm,
        "Training Data"
    )

    print("\n" + "=" * 60)
    print("EVALUATING ON SHARD 2 TEST DATA")
    print("=" * 60)

    s2_rule, s2_single, s2_multi = evaluate_all_systems(
        shard2_spike,
        lstm,
        "Shard 2 Test"
    )

    plot_cross_shard_comparison(
        s0_rule, s0_single, s0_multi,
        s2_rule, s2_single, s2_multi,
        shard0_spike, shard2_spike
    )

    print_generalization_verdict(
        s0_rule, s0_single, s0_multi,
        s2_rule, s2_single, s2_multi
    )

    print("\nVALIDATION COMPLETE")


if __name__ == "__main__":
    main()