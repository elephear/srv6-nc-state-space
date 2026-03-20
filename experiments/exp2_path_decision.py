"""
Experiment 2: Path Decision Impact Verification.

Validates that real-time updates to the joint state tensor S effectively
influence path computation decisions by CPM.

Scenario:
  1. U1 sends a computing request to center node C1.
  2. Initial state: all links low utilization (<30%), all CPS GPU load 20-40%.
     Optimal path: U1 -> R31 -> R32 -> R33 -> C1
  3. Inject background traffic on R32-R33 (utilization -> 90%).
  4. Send tasks to E5 (GPU load -> 80%).
  5. Re-request path: should reroute to avoid congested link.
     New path: U1 -> R31 -> R41 -> R42 -> R33 -> C1 (or similar)

Expected result:
  CPM recalculates path to avoid congested link R32-R33,
  demonstrating that S updates drive path optimization.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.cpm import CPM
from modules.state_tensor import NET_DELAY, NET_UTILIZATION
from utils.topology_builder import build_full_simulation


def run_experiment_2(output_dir: str = "results/exp2"):
    """Run Experiment 2: Path Decision Impact.

    Args:
        output_dir: Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Experiment 2: Path Decision Impact Verification")
    print("=" * 60)

    # Step 1: Build simulation environment
    print("\n[Step 1] Building simulation environment...")
    sim = build_full_simulation()
    graph = sim["graph"]
    tensor = sim["tensor"]
    cpr = sim["cpr"]

    cpm = CPM(tensor=tensor, cpr=cpr, graph=graph)

    # Step 2: Set initial conditions - low utilization everywhere
    print("\n[Step 2] Setting initial conditions (low utilization)...")

    for src, dst in graph.edges():
        tensor.update_link_state(
            src, dst,
            bandwidth=0.9,
            delay=0.1,
            jitter=0.02,
            utilization=0.15,  # ~15% utilization
            queue_length=0.05,
        )

    # Set moderate GPU load (20-40%) for all compute nodes
    compute_nodes = sim["edge_compute_ids"] + sim["center_compute_ids"]
    for nid in compute_nodes:
        gpu_load = np.random.uniform(0.2, 0.4)
        cpr.update_cps(nid, gpu=gpu_load, cpu=gpu_load * 0.5, task_queue=0.1)

    # Sync to tensor
    cpm.sync_compute_state()

    # Step 3: Initial path request - U1 to C1
    print("\n[Step 3] Initial path request: U1 -> C1...")

    result_initial = cpm.path_request(src="U1", dst="C1")
    if result_initial:
        target, sid_list, cost = result_initial
        initial_path = ["U1"] + sid_list
        print(f"  Target: {target}")
        print(f"  Path: {' -> '.join(initial_path)}")
        print(f"  Cost: {cost:.4f}")
    else:
        print("  ERROR: No path found!")
        initial_path = []

    # Step 4: Inject congestion on R32-R33
    print("\n[Step 4] Injecting congestion on R32->R33 (util: 15% -> 90%)...")

    cpm.update_link_state("R32", "R33", utilization=0.90, delay=0.5, jitter=0.3)
    # Also update reverse direction
    cpm.update_link_state("R33", "R32", utilization=0.85, delay=0.45, jitter=0.25)

    # Verify tensor update
    r32_r33_state = tensor.get_link_state("R32", "R33")
    print(f"  R32->R33 utilization in S: {r32_r33_state[NET_UTILIZATION]:.4f}")
    print(f"  R32->R33 delay in S: {r32_r33_state[NET_DELAY]:.4f}")

    # Step 5: Increase E5 load
    print("\n[Step 5] Increasing E5 GPU load (sending tasks)...")

    for i in range(5):
        cpr.simulate_task_arrival("E5", gpu_increase=0.12, queue_increase=3)

    e5_state = cpr.get_node_state("E5")
    print(f"  E5 GPU utilization: {e5_state[0]:.4f}")
    print(f"  E5 task queue: {e5_state[2]:.4f}")

    # Sync
    cpm.sync_compute_state()

    # Step 6: Re-request path with updated state
    print("\n[Step 6] Re-requesting path: U1 -> C1 (after congestion)...")

    result_updated = cpm.path_request(src="U1", dst="C1")
    if result_updated:
        target, sid_list, cost = result_updated
        updated_path = ["U1"] + sid_list
        print(f"  Target: {target}")
        print(f"  Path: {' -> '.join(updated_path)}")
        print(f"  Cost: {cost:.4f}")
    else:
        print("  ERROR: No path found!")
        updated_path = []

    # Step 7: Verify path changed
    print("\n[Step 7] Comparing paths...")

    path_changed = initial_path != updated_path
    avoids_congested_link = not (
        "R32" in updated_path
        and "R33" in updated_path
        and updated_path.index("R33") == updated_path.index("R32") + 1
    )

    print(f"  Initial path: {' -> '.join(initial_path)}")
    print(f"  Updated path: {' -> '.join(updated_path)}")
    print(f"  Path changed: {path_changed}")
    print(f"  Avoids R32->R33: {avoids_congested_link}")

    # Step 8: Also test with different compute node selection
    print("\n[Step 8] Testing compute node selection (no fixed dst)...")

    # Make C1 heavily loaded
    cpr.update_cps("C1", gpu=0.85, cpu=0.7, task_queue=0.8)
    # Keep C2 lightly loaded
    cpr.update_cps("C2", gpu=0.2, cpu=0.15, task_queue=0.1)
    cpm.sync_compute_state()

    result_auto = cpm.path_request(src="U1")
    if result_auto:
        auto_target, auto_sid_list, auto_cost = result_auto
        auto_path = ["U1"] + auto_sid_list
        print(f"  Auto-selected target: {auto_target}")
        print(f"  Path: {' -> '.join(auto_path)}")
        print(f"  Cost: {auto_cost:.4f}")
        prefers_c2 = auto_target == "C2"
        print(f"  Prefers C2 (lower load): {prefers_c2}")
    else:
        print("  No path found for auto-selection!")
        auto_target = None
        auto_path = []
        prefers_c2 = False

    # Step 9: Save results
    print("\n[Step 9] Saving results...")

    results = {
        "experiment": "exp2_path_decision_impact",
        "initial_conditions": {
            "link_utilization": 0.15,
            "gpu_load_range": [0.2, 0.4],
        },
        "initial_path": {
            "path": initial_path,
            "cost": result_initial[2] if result_initial else None,
        },
        "congestion_event": {
            "link": "R32->R33",
            "new_utilization": 0.90,
            "e5_gpu_load": float(e5_state[0]),
        },
        "updated_path": {
            "path": updated_path,
            "cost": result_updated[2] if result_updated else None,
        },
        "auto_selection": {
            "target": auto_target,
            "path": auto_path,
            "cost": result_auto[2] if result_auto else None,
        },
        "verification": {
            "path_changed": path_changed,
            "avoids_congested_link": avoids_congested_link,
            "prefers_lower_load_node": prefers_c2,
        },
    }

    results_path = os.path.join(output_dir, "exp2_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Experiment 2 Results Summary")
    print("=" * 60)
    all_pass = path_changed and avoids_congested_link
    print(f"  Path rerouting on congestion:  {'PASS' if path_changed else 'FAIL'}")
    print(f"  Avoids congested link:         {'PASS' if avoids_congested_link else 'FAIL'}")
    print(f"  Compute-aware node selection:  {'PASS' if prefers_c2 else 'FAIL'}")
    print(f"  Overall:                       {'PASS' if all_pass else 'FAIL'}")

    return results


if __name__ == "__main__":
    results = run_experiment_2()
