"""
Topology Builder Utility.

Constructs the 4x4 grid backbone topology with edge/center compute nodes
and user access nodes as described in the paper (Figure 5).

Topology:
  - 16 core routers: R11-R44 (4x4 grid)
  - 8 edge compute nodes: E1-E8 (connected to R11-R24)
  - 2 center compute nodes: C1, C2 (connected to R33, R34)
  - 4 user nodes: U1-U4 (connected to R31, R32, R41, R42)

Link parameters:
  - Bandwidth: 100 Mbps (normalized to 1.0)
  - Propagation delay: 2 ms (normalized based on max expected ~20ms)
"""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from modules.cpr import CPR
from modules.state_tensor import JointStateTensor


def get_router_id(row: int, col: int) -> str:
    """Get router ID from grid position (1-indexed)."""
    return f"R{row}{col}"


def build_topology() -> Tuple[
    nx.DiGraph,
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """Build the simulation topology graph.

    Returns:
        Tuple of (graph, all_node_ids, router_ids, edge_compute_ids,
                  center_compute_ids, user_ids).
    """
    G = nx.DiGraph()

    # --- Core routers: 4x4 grid ---
    router_ids = []
    for row in range(1, 5):
        for col in range(1, 5):
            rid = get_router_id(row, col)
            router_ids.append(rid)
            G.add_node(rid, type="router", row=row, col=col)

    # Grid links (bidirectional): horizontal and vertical
    for row in range(1, 5):
        for col in range(1, 5):
            src = get_router_id(row, col)
            # Right neighbor
            if col < 4:
                dst = get_router_id(row, col + 1)
                G.add_edge(src, dst)
                G.add_edge(dst, src)
            # Down neighbor
            if row < 4:
                dst = get_router_id(row + 1, col)
                G.add_edge(src, dst)
                G.add_edge(dst, src)

    # --- Edge compute nodes: E1-E8 connected to R11-R24 ---
    edge_compute_ids = [f"E{i}" for i in range(1, 9)]
    # E1-E4 connect to R11-R14 (row 1)
    # E5-E8 connect to R21-R24 (row 2)
    edge_connections = {
        "E1": "R11", "E2": "R12", "E3": "R13", "E4": "R14",
        "E5": "R21", "E6": "R22", "E7": "R23", "E8": "R24",
    }
    for eid, rid in edge_connections.items():
        G.add_node(eid, type="edge_compute")
        G.add_edge(eid, rid)
        G.add_edge(rid, eid)

    # --- Center compute nodes: C1, C2 ---
    center_compute_ids = ["C1", "C2"]
    center_connections = {"C1": "R33", "C2": "R34"}
    for cid, rid in center_connections.items():
        G.add_node(cid, type="center_compute")
        G.add_edge(cid, rid)
        G.add_edge(rid, cid)

    # --- User nodes: U1-U4 ---
    user_ids = ["U1", "U2", "U3", "U4"]
    user_connections = {
        "U1": "R31", "U2": "R32", "U3": "R41", "U4": "R42",
    }
    for uid, rid in user_connections.items():
        G.add_node(uid, type="user")
        G.add_edge(uid, rid)
        G.add_edge(rid, uid)

    all_node_ids = router_ids + edge_compute_ids + center_compute_ids + user_ids

    return (
        G,
        all_node_ids,
        router_ids,
        edge_compute_ids,
        center_compute_ids,
        user_ids,
    )


def initialize_state_tensor(
    graph: nx.DiGraph,
    all_node_ids: List[str],
    compute_node_ids: List[str],
    base_bandwidth: float = 1.0,
    base_delay: float = 0.1,
    base_jitter: float = 0.02,
    base_utilization: float = 0.1,
    base_queue: float = 0.05,
) -> JointStateTensor:
    """Initialize the joint state tensor with default values.

    Args:
        graph: Network topology graph.
        all_node_ids: All node IDs.
        compute_node_ids: Computing node IDs.
        base_bandwidth: Initial bandwidth (normalized).
        base_delay: Initial delay (normalized, 2ms/20ms = 0.1).
        base_jitter: Initial jitter.
        base_utilization: Initial link utilization.
        base_queue: Initial queue length.

    Returns:
        Initialized JointStateTensor.
    """
    tensor = JointStateTensor(all_node_ids, compute_node_ids)

    # Initialize link states
    for src, dst in graph.edges():
        tensor.add_edge(src, dst)
        # Add small random perturbation
        noise = np.random.uniform(-0.02, 0.02, 5)
        state = np.array([
            base_bandwidth + noise[0],
            base_delay + noise[1],
            base_jitter + noise[2],
            base_utilization + noise[3],
            base_queue + noise[4],
        ])
        state = np.clip(state, 0.0, 1.0)
        tensor.set_link_state(src, dst, state)

    return tensor


def initialize_cpr(
    edge_compute_ids: List[str],
    center_compute_ids: List[str],
) -> CPR:
    """Initialize CPR with all computing nodes registered.

    Args:
        edge_compute_ids: Edge computing node IDs.
        center_compute_ids: Center computing node IDs.

    Returns:
        Initialized CPR instance.
    """
    cpr = CPR()

    # Register edge nodes
    for eid in edge_compute_ids:
        initial_state = np.array([
            np.random.uniform(0.1, 0.3),   # GPU utilization (low initially)
            np.random.uniform(0.1, 0.3),   # CPU load
            0.0,                            # Task queue (empty)
            np.random.uniform(0.05, 0.15),  # Energy weight
            1.0,                            # Reachability (fully reachable)
        ])
        cpr.register_node(
            eid,
            node_type="edge",
            service_types=["inference", "container"],
            initial_state=initial_state,
        )

    # Register center nodes (higher capacity)
    for cid in center_compute_ids:
        initial_state = np.array([
            np.random.uniform(0.1, 0.4),   # GPU utilization
            np.random.uniform(0.1, 0.3),   # CPU load
            0.0,                            # Task queue
            np.random.uniform(0.08, 0.2),  # Energy weight
            1.0,                            # Reachability
        ])
        cpr.register_node(
            cid,
            node_type="center",
            service_types=["inference", "training", "container", "vm"],
            initial_state=initial_state,
        )

    return cpr


def build_full_simulation():
    """Build and return all simulation components.

    Returns:
        Dict with keys: graph, tensor, cpr, node_ids, router_ids,
        edge_compute_ids, center_compute_ids, user_ids.
    """
    (
        graph,
        all_node_ids,
        router_ids,
        edge_compute_ids,
        center_compute_ids,
        user_ids,
    ) = build_topology()

    compute_node_ids = edge_compute_ids + center_compute_ids

    tensor = initialize_state_tensor(
        graph, all_node_ids, compute_node_ids
    )

    cpr = initialize_cpr(edge_compute_ids, center_compute_ids)

    # Sync CPR state into tensor
    for nid in compute_node_ids:
        state = cpr.get_node_state(nid)
        tensor.set_compute_state(nid, state)

    return {
        "graph": graph,
        "tensor": tensor,
        "cpr": cpr,
        "all_node_ids": all_node_ids,
        "router_ids": router_ids,
        "edge_compute_ids": edge_compute_ids,
        "center_compute_ids": center_compute_ids,
        "user_ids": user_ids,
    }
