# Multi-Agent Predictive RL Cloud Auto-Scaler

A cloud resource autoscaling system that combines **LSTM-based traffic prediction**, **Multi-Agent Reinforcement Learning**, and **Anomaly Detection** to proactively scale virtual machines before traffic spikes occur.

Trained and validated on real **Google Cluster Trace 2019** data — 400,000 genuine CPU usage records from Google's production datacenters (May 2019).

---

## Architecture

```
Google Cluster Trace 2019 (Real Data)
              ↓
     LSTM Workload Predictor
     (predicts next traffic value)
              ↓
     Anomaly Detector
     (Isolation Forest — catches extreme patterns)
              ↓
     Multi-Agent RL Auto-Scaler
     (3 PPO agents, one per cluster)
              ↓
     Cloud Environment Simulator
     (Gymnasium-compatible)
              ↓
     Scaling Decisions (VM up/down/hold)
```

---

## Key Results

Evaluated on **Shard 2** — data the models never saw during training:

| System | SLA Violations | Cost | Scaling Actions |
|---|---|---|---|
| Rule-Based | 22 | 936 | 41 |
| Single RL Agent | 18 | 1320 | 15 |
| **Multi-Agent RL** | **0** | **1000** | **18** |

- **Zero SLA violations** on training data
- **15% lower cost** than rule-based on unseen test data
- LSTM prediction error: **8.45%** on spike traffic
- Validated with **80/20 train/validation split** — no overfitting detected

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Neural Network | PyTorch (LSTM) |
| Reinforcement Learning | Stable-Baselines3 (PPO) |
| RL Environment | Gymnasium |
| Anomaly Detection | Scikit-learn (Isolation Forest) |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Data Source | Google Cloud SDK / gsutil |

---

## Project Structure

```
cloud-autoscaler/
│
├── src/
│   ├── cloud_env.py          # Gymnasium cloud simulator
│   ├── lstm_predictor.py     # PyTorch LSTM model
│   ├── anomaly_detector.py   # Isolation Forest detector
│   ├── rl_agent.py           # PPO training & evaluation
│   ├── multi_agent_scaler.py # 3-agent system
│   └── workload_generator.py # Synthetic backup generator
│
├── dashboard/
│   └── dashboard.py          # Streamlit live demo
│
├── data/
│   ├── workload_normal.npy   # Training workload (Shard 0+1)
│   ├── workload_spike.npy    # Training workload (Shard 0+1)
│   ├── workload_wave.npy     # Training workload (Shard 0+1)
│   └── test/
│       ├── workload_normal.npy  # Test workload (Shard 2)
│       ├── workload_spike.npy   # Test workload (Shard 2)
│       └── workload_wave.npy    # Test workload (Shard 2)
│
├── models/                   # Trained model files (see Setup)
│   ├── lstm_model.pth
│   ├── ppo_spike_agent.zip
│   ├── multi_agent_1.zip
│   ├── multi_agent_2.zip
│   └── multi_agent_3.zip
│
├── results/                  # Generated graphs
│
├── train.py                  # Master training script
├── demo.py                   # Terminal demo
├── data_loader.py            # Google trace processor
├── validate_lstm.py          # LSTM validation suite
├── validate_agents.py        # Agent generalization test
├── test_env.py               # Environment smoke test
└── requirements.txt
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOURNAME/cloud-autoscaler.git
cd cloud-autoscaler
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the data

Download the data folder from the shared Google Drive link and place it in the project root. Your folder structure should look like:

```
cloud-autoscaler/
└── data/
    ├── workload_normal.npy
    ├── workload_spike.npy
    ├── workload_wave.npy
    ├── test/
    │   ├── workload_normal.npy
    │   ├── workload_spike.npy
    │   └── workload_wave.npy
    └── raw/
        ├── instance_usage-000000000000.json.gz  ← Shard 0
        ├── instance_usage-000000000001.json.gz  ← Shard 1
        └── instance_usage-000000000002.json.gz  ← Shard 2
```

> The raw files are genuine Google Cluster Trace 2019 data downloaded from `gs://clusterdata_2019_a/`. Shards 0+1 were used for training. Shard 2 was reserved exclusively for testing.

---

## Running The Project

### Quick demo (loads pre-trained models, shows results)

```bash
python demo.py
```

### Launch live dashboard

```bash
python -m streamlit run dashboard/dashboard.py
```

Then open your browser at `http://localhost:8501`

### Verify environment works

```bash
python test_env.py
```

---

## Training From Scratch

> **Note:** Pre-trained models are included. Only run this if you want to retrain everything. Takes ~2 hours on CPU.

```bash
# Step 1 — Process training data (Shard 0 + Shard 1)
python data_loader.py --process

# Step 2 — Process test data (Shard 2 — never train on this)
python data_loader.py --process-test

# Step 3 — Verify all data is correct
python data_loader.py --verify

# Step 4 — Train everything (LSTM + PPO + Multi-Agent)
python train.py
```

---

## Validation

### Check LSTM for overfitting

```bash
python validate_lstm.py
```

Output includes:
- Train vs validation loss curves
- Prediction accuracy on Shard 0 (training data)
- Generalization test on Shard 1 (unseen data)

### Test agent generalization on Shard 2

```bash
python validate_agents.py
```

Output includes:
- Side by side comparison: Training data vs Shard 2
- SLA violations, cost, scaling actions for all three systems
- Generalization verdict

---

## How It Works

### 1. Cloud Environment Simulator (`src/cloud_env.py`)
Simulates a cloud infrastructure with VMs, CPU utilization, request queues and response time. Built on the OpenAI Gymnasium standard so any RL algorithm can plug in.

### 2. LSTM Predictor (`src/lstm_predictor.py`)
A 2-layer LSTM neural network trained on real Google datacenter traffic. Takes the last 10 traffic readings and predicts the next value — giving RL agents a 1-step lookahead before traffic arrives.

### 3. Anomaly Detector (`src/anomaly_detector.py`)
Isolation Forest trained on normal traffic patterns. When it detects extreme traffic the RL agent has never seen, it bypasses the agent entirely and triggers emergency scaling.

### 4. Multi-Agent RL (`src/multi_agent_scaler.py`)
Three independent PPO agents, each managing their own 333-step cluster. Each agent sees: current CPU, VM count, queue length, response time, and the LSTM prediction. Trained for 200,000 timesteps each.

### 5. Dashboard (`dashboard/dashboard.py`)
Streamlit web app showing live animated comparison of all three systems responding to the same traffic in real time.

---

## Data Source

**Google Cluster Trace 2019**
- Source: `gs://clusterdata_2019_a/` (Google Cloud Storage — publicly accessible)
- Format: JSON.gz, one CPU usage record per line
- Records used: 400,000 (Shard 0 + Shard 1 for training, Shard 2 for testing)
- Field used: `average_usage.cpus` — average CPU utilization per machine per interval
- Conversion: CPU utilization × 400 = requests/min

To download additional shards directly:
```bash
gsutil cp gs://clusterdata_2019_a/instance_usage-000000000003.json.gz data/raw/
```

---

## Why PPO Over DQN

The original project plan specified DQN. We switched to PPO because:
- PPO's clip mechanism prevents catastrophic policy updates — critical for stable cloud scaling
- More stable training on continuous state spaces
- Industry standard since 2017 — used by OpenAI, DeepMind, and major cloud providers
- Our convergence curve confirmed smooth training from -5000 to +4300 reward

---

## Limitations & Future Work

**Current limitation:** Multi-Agent performance drops on Shard 2 (SLA violations increase) because spike timing differs from training data. Each agent specializes in a fixed time window — when spikes occur outside that window the agent isn't prepared.

**Future improvements:**
1. Dynamic cluster assignment based on detected traffic patterns
2. Train on more diverse spike timings across shards
3. Transfer learning for faster adaptation to new datasets
4. Online learning — continuous policy updates on live traffic
5. Deploy on actual cloud infrastructure (AWS/GCP) instead of simulator

---

## Team

Built as a Final Year Computer Science Engineering project.

---

## References

- Reiss et al. (2012) — Google Cluster Usage Traces
- Cortez et al. (2017) — Resource Central, Microsoft Azure
- Liu et al. (2020) — DRL-based Cloud Autoscaling
- Schulman et al. (2017) — Proximal Policy Optimization
- Liu et al. (2008) — Isolation Forest