# srv6-nc-state-space

面向算力网络的联合状态感知调度仿真平台。实现基于 P4 的 FCLS 协议扩展、ONOS 控制器协同及 CPR 联合状态张量维护，支持边缘 AI 推理与算力跨域调度两类业务，为 SRv6 算力感知调度提供可复现的实验基础。

## Project Structure

```
srv6-nc-state-space/
├── modules/                    # Core simulation modules
│   ├── state_tensor.py         # Joint state tensor S ∈ R^{|V|x|V|x5}
│   ├── cpr.py                  # Computing Power Registry (CPR)
│   ├── cps.py                  # Computing Power Service node simulator
│   └── cpm.py                  # Controller Policy Module (path computation)
├── p4/
│   └── fcls_switch.p4          # P4 program: FCLS protocol + SRv6 forwarding
├── topology/
│   └── topo.py                 # Mininet topology (requires Mininet + BMv2)
├── utils/
│   └── topology_builder.py     # Standalone topology builder (no Mininet needed)
├── experiments/
│   ├── exp1_state_update.py    # Experiment 1: State update correctness
│   ├── exp2_path_decision.py   # Experiment 2: Path decision impact
│   └── plot_results.py         # Paper-ready figure generation
├── tests/
│   ├── test_tensor.py          # Joint state tensor tests
│   ├── test_cpr.py             # CPR module tests
│   └── test_cpm.py             # CPM module tests
└── requirements.txt
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run tests

```bash
python -m pytest tests/ -v
```

### Run experiments

```bash
# Experiment 1: State Update Correctness Verification
python experiments/exp1_state_update.py

# Experiment 2: Path Decision Impact Verification
python experiments/exp2_path_decision.py

# Generate paper figures
python experiments/plot_results.py
```

Results are saved to `results/exp1/` and `results/exp2/`.

## Architecture

### Joint State Tensor

The tensor `S ∈ R^{|V|x|V|x5}` unifies network and computing states:

- **Link state** `S[i,j]` = `[bandwidth, delay, jitter, utilization, queue_length]`
- **Compute state** `S[n,n]` = `[GPU_util, CPU_load, task_queue, energy_weight, reachability]`
- **Other entries** = zero vectors

### Simulation Topology

4x4 grid backbone (R11-R44) with:
- 8 edge compute nodes (E1-E8) on rows 1-2
- 2 center compute nodes (C1, C2) on row 3
- 4 user nodes (U1-U4) on rows 3-4

### Key Modules

| Module | Description |
|--------|-------------|
| `state_tensor.py` | Maintains the joint state tensor S with link and compute state operations |
| `cpr.py` | Computing Power Registry -- node registration, state reporting, EWMA reachability |
| `cps.py` | CPS node simulator -- task queuing, resource consumption modeling |
| `cpm.py` | Controller Policy Module -- weighted shortest path, SRv6 SID list generation |
| `fcls_switch.p4` | P4 data plane -- FCLS parsing, flow cache, SRv6 forwarding tables |

## Experiments

### Experiment 1: State Update Correctness

Validates that S correctly captures dynamic changes. Injects a link utilization spike (R12-R22: 10% -> 70%) and verifies the tensor updates within the polling interval.

### Experiment 2: Path Decision Impact

Shows that S updates drive path rerouting. Initial path U1->R31->R32->R33->C1 is rerouted to avoid congested link R32-R33 after traffic injection.
