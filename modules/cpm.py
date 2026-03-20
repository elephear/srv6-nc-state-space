"""
Controller Policy Module (CPM).

Northbound application running on ONOS controller. CPM is responsible for:
  1. Collecting network state from ONOS REST API (link bandwidth, delay, etc.)
  2. Collecting computing state from CPR (GPU, CPU, queue, energy, reachability)
  3. Maintaining the global joint state tensor S
  4. Running scheduling algorithms to compute optimal paths
  5. Returning SRv6 label stacks (SID lists) for path requests

The scheduling algorithm used here is a weighted shortest path that combines
link cost (delay + utilization) with node cost (1/GPU_available), serving as
a baseline. In production, this can be replaced by GCN-DQN or other ML models.
"""

import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from modules.cpr import CPR
from modules.state_tensor import (
    COMP_GPU,
    COMP_REACHABILITY,
    COMP_TASK_QUEUE,
    JointStateTensor,
    NET_DELAY,
    NET_UTILIZATION,
    STATE_DIM,
)


class CPM:
    """Controller Policy Module.

    Maintains the joint state tensor and provides path computation
    services for computing-aware scheduling.
    """

    def __init__(
        self,
        tensor: JointStateTensor,
        cpr: CPR,
        graph: nx.DiGraph,
        delay_weight: float = 0.4,
        util_weight: float = 0.3,
        gpu_weight: float = 0.2,
        queue_weight: float = 0.1,
        reachability_threshold: float = 0.3,
    ):
        """Initialize CPM.

        Args:
            tensor: The joint state tensor to maintain.
            cpr: Reference to the CPR module.
            graph: NetworkX directed graph of the topology.
            delay_weight: Weight for delay in link cost.
            util_weight: Weight for utilization in link cost.
            gpu_weight: Weight for GPU load in node cost.
            queue_weight: Weight for task queue in node cost.
            reachability_threshold: Minimum reachability to consider a node.
        """
        self.tensor = tensor
        self.cpr = cpr
        self.graph = graph
        self.delay_weight = delay_weight
        self.util_weight = util_weight
        self.gpu_weight = gpu_weight
        self.queue_weight = queue_weight
        self.reachability_threshold = reachability_threshold

        # Update tracking
        self._last_network_update = 0.0
        self._last_compute_update = 0.0
        self._update_history: List[Dict] = []

    def sync_compute_state(self) -> None:
        """Sync computing node states from CPR into the tensor.

        Calls CPR.get_cps() and updates diagonal elements of S.
        """
        states = self.cpr.get_cps()
        for node_id, state_vec in states.items():
            self.tensor.set_compute_state(node_id, state_vec)
        self._last_compute_update = time.time()

    def update_link_state(
        self,
        src: str,
        dst: str,
        bandwidth: Optional[float] = None,
        delay: Optional[float] = None,
        jitter: Optional[float] = None,
        utilization: Optional[float] = None,
        queue_length: Optional[float] = None,
    ) -> None:
        """Update a single link's state in the tensor.

        In production, this data comes from ONOS REST API.
        """
        self.tensor.update_link_state(
            src, dst, bandwidth, delay, jitter, utilization, queue_length
        )
        self._last_network_update = time.time()

    def update_all_link_states(
        self, link_states: Dict[Tuple[str, str], np.ndarray]
    ) -> None:
        """Batch update all link states.

        Args:
            link_states: Dict mapping (src, dst) to 5D state vectors.
        """
        for (src, dst), state_vec in link_states.items():
            self.tensor.set_link_state(src, dst, state_vec)
        self._last_network_update = time.time()
        self._update_history.append({
            "timestamp": self._last_network_update,
            "type": "network_batch",
            "count": len(link_states),
        })

    def _compute_link_cost(self, src: str, dst: str) -> float:
        """Compute the cost of traversing a link.

        Cost = delay_weight * D + util_weight * U

        Args:
            src: Source node.
            dst: Destination node.

        Returns:
            Link cost (lower is better).
        """
        state = self.tensor.get_link_state(src, dst)
        delay = state[NET_DELAY]
        util = state[NET_UTILIZATION]
        return self.delay_weight * delay + self.util_weight * util

    def _compute_node_cost(self, node_id: str) -> float:
        """Compute the cost of using a computing node.

        Cost = gpu_weight * G + queue_weight * Q^task

        Lower GPU utilization and shorter queue = lower cost.

        Args:
            node_id: Computing node ID.

        Returns:
            Node cost (lower is better).
        """
        state = self.tensor.get_compute_state(node_id)
        gpu = state[COMP_GPU]
        queue = state[COMP_TASK_QUEUE]
        return self.gpu_weight * gpu + self.queue_weight * queue

    def _get_eligible_compute_nodes(
        self, service_type: Optional[str] = None
    ) -> List[str]:
        """Get computing nodes that meet reachability threshold.

        Args:
            service_type: Optional filter by service type.

        Returns:
            List of eligible node IDs.
        """
        if service_type:
            candidates = self.cpr.get_nodes_by_service(service_type)
        else:
            candidates = self.cpr.get_cps()

        eligible = []
        for node_id, state in candidates.items():
            if state[COMP_REACHABILITY] >= self.reachability_threshold:
                eligible.append(node_id)
        return eligible

    def path_request(
        self,
        src: str,
        task_type: Optional[str] = None,
        dst: Optional[str] = None,
    ) -> Optional[Tuple[str, List[str], float]]:
        """Compute the optimal path from source to best computing node.

        Uses weighted shortest path algorithm combining link costs and
        node costs to find the best (path, target_node) pair.

        Args:
            src: Source node ID (user node or switch).
            task_type: Required service type for filtering compute nodes.
            dst: Optional specific destination (if not provided, finds best).

        Returns:
            Tuple of (target_node_id, sid_list, total_cost) or None if
            no path found.
        """
        # Sync latest compute state
        self.sync_compute_state()

        # Build weighted graph for path computation
        weighted_graph = nx.DiGraph()

        for u, v in self.graph.edges():
            cost = self._compute_link_cost(u, v)
            weighted_graph.add_edge(u, v, weight=cost)

        if dst:
            # Direct path to specific destination
            target_nodes = [dst]
        else:
            # Find eligible computing nodes
            target_nodes = self._get_eligible_compute_nodes(task_type)

        if not target_nodes:
            return None

        best_path = None
        best_target = None
        best_cost = float("inf")

        for target in target_nodes:
            try:
                path = nx.shortest_path(
                    weighted_graph, src, target, weight="weight"
                )
                path_cost = nx.shortest_path_length(
                    weighted_graph, src, target, weight="weight"
                )
                # Add node cost for the target computing node
                if target in self.tensor.compute_node_ids:
                    node_cost = self._compute_node_cost(target)
                    total_cost = path_cost + node_cost
                else:
                    total_cost = path_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = path
                    best_target = target
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        if best_path is None:
            return None

        # Convert path to SID list (SRv6 segment identifiers)
        sid_list = self._path_to_sid_list(best_path)

        return (best_target, sid_list, best_cost)

    def _path_to_sid_list(self, path: List[str]) -> List[str]:
        """Convert a node path to an SRv6 SID list.

        In a real SRv6 deployment, each node has an IPv6 SID address.
        Here we use node IDs as SID placeholders.

        Args:
            path: Ordered list of node IDs.

        Returns:
            SID list for SRv6 header.
        """
        # Each intermediate and destination node becomes a segment
        return list(path[1:])  # Exclude source node

    def get_tensor_snapshot(self) -> np.ndarray:
        """Get a snapshot of the current joint state tensor."""
        return self.tensor.get_snapshot()

    def get_update_history(self) -> List[Dict]:
        """Get the history of state updates."""
        return list(self._update_history)

    def get_all_link_costs(self) -> Dict[Tuple[str, str], float]:
        """Get computed link costs for all edges."""
        costs = {}
        for u, v in self.graph.edges():
            costs[(u, v)] = self._compute_link_cost(u, v)
        return costs

    def get_all_node_costs(self) -> Dict[str, float]:
        """Get computed node costs for all computing nodes."""
        costs = {}
        for nid in self.tensor.compute_node_ids:
            costs[nid] = self._compute_node_cost(nid)
        return costs

    def __repr__(self) -> str:
        return (
            f"CPM(nodes={self.tensor.n}, "
            f"compute_nodes={len(self.tensor.compute_node_ids)}, "
            f"edges={self.graph.number_of_edges()})"
        )
