"""
Computing Power Registry (CPR) Module.

Maintains the registry and real-time state of all computing power service
nodes (CPS). Each CPS node reports its 5D state vector periodically:
  [G, L, Q^task, P, R]
  - G: GPU utilization intensity
  - L: CPU load
  - Q^task: Task queue length
  - P: Energy weight per unit task
  - R: Dynamic reachability

CPR provides two core interfaces:
  - get_cps(): Returns current state of all computing nodes.
  - update_cps(node_id, new_state): Updates a specific node's state.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ComputeNodeInfo:
    """Registration and state information for a computing node."""

    node_id: str
    node_type: str  # "edge" or "center"
    service_types: List[str] = field(default_factory=list)
    # 5D state: [G, L, Q^task, P, R]
    gpu_utilization: float = 0.0
    cpu_load: float = 0.0
    task_queue_length: float = 0.0
    energy_weight: float = 0.0
    reachability: float = 1.0
    last_update: float = 0.0
    registered: bool = False

    def to_state_vector(self) -> np.ndarray:
        """Convert to 5D state vector [G, L, Q^task, P, R]."""
        return np.array([
            self.gpu_utilization,
            self.cpu_load,
            self.task_queue_length,
            self.energy_weight,
            self.reachability,
        ])

    def from_state_vector(self, vec: np.ndarray) -> None:
        """Update state from a 5D vector."""
        self.gpu_utilization = float(vec[0])
        self.cpu_load = float(vec[1])
        self.task_queue_length = float(vec[2])
        self.energy_weight = float(vec[3])
        self.reachability = float(vec[4])
        self.last_update = time.time()


class CPR:
    """Computing Power Registry.

    Central registry for all computing power service nodes. CPS nodes
    register on startup and periodically report their state.
    """

    def __init__(self, reachability_alpha: float = 0.2):
        """Initialize CPR.

        Args:
            reachability_alpha: Smoothing factor for reachability updates.
        """
        self._nodes: Dict[str, ComputeNodeInfo] = {}
        self._lock = threading.Lock()
        self._alpha = reachability_alpha
        self._history: Dict[str, List[Dict]] = {}

    def register_node(
        self,
        node_id: str,
        node_type: str = "edge",
        service_types: Optional[List[str]] = None,
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        """Register a new computing node.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type of node ("edge" or "center").
            service_types: List of supported service types.
            initial_state: Optional initial 5D state vector.
        """
        with self._lock:
            info = ComputeNodeInfo(
                node_id=node_id,
                node_type=node_type,
                service_types=service_types or [],
                registered=True,
                last_update=time.time(),
            )
            if initial_state is not None:
                info.from_state_vector(initial_state)
            self._nodes[node_id] = info
            self._history[node_id] = []

    def unregister_node(self, node_id: str) -> None:
        """Unregister a computing node."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].registered = False

    def update_cps(
        self,
        node_id: str,
        new_state: Optional[np.ndarray] = None,
        gpu: Optional[float] = None,
        cpu: Optional[float] = None,
        task_queue: Optional[float] = None,
        energy: Optional[float] = None,
        probe_result: Optional[int] = None,
    ) -> None:
        """Update the state of a computing node.

        Can update with a full state vector or individual fields.

        Args:
            node_id: Node identifier.
            new_state: Full 5D state vector (overrides individual fields).
            gpu: GPU utilization [0,1].
            cpu: CPU load [0,1].
            task_queue: Task queue length (normalized).
            energy: Energy weight.
            probe_result: Probe result for reachability (0 or 1).
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' is not registered")

            node = self._nodes[node_id]

            if new_state is not None:
                node.from_state_vector(new_state)
            else:
                if gpu is not None:
                    node.gpu_utilization = np.clip(gpu, 0.0, 1.0)
                if cpu is not None:
                    node.cpu_load = np.clip(cpu, 0.0, 1.0)
                if task_queue is not None:
                    node.task_queue_length = max(0.0, task_queue)
                if energy is not None:
                    node.energy_weight = max(0.0, energy)
                if probe_result is not None:
                    # Exponential moving average: R(t) = alpha*probe + (1-alpha)*R(t-1)
                    node.reachability = (
                        self._alpha * probe_result
                        + (1 - self._alpha) * node.reachability
                    )
                node.last_update = time.time()

            # Record history for experiment tracking
            self._history[node_id].append({
                "timestamp": node.last_update,
                "state": node.to_state_vector().tolist(),
            })

    def get_cps(self) -> Dict[str, np.ndarray]:
        """Get current state of all registered computing nodes.

        Returns:
            Dict mapping node_id to 5D state vector.
        """
        with self._lock:
            return {
                nid: node.to_state_vector()
                for nid, node in self._nodes.items()
                if node.registered
            }

    def get_node_state(self, node_id: str) -> np.ndarray:
        """Get state of a specific computing node.

        Returns:
            5D state vector [G, L, Q^task, P, R].
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' is not registered")
            return self._nodes[node_id].to_state_vector()

    def get_node_info(self, node_id: str) -> ComputeNodeInfo:
        """Get full info for a computing node."""
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' is not registered")
            return self._nodes[node_id]

    def get_nodes_by_type(self, node_type: str) -> Dict[str, np.ndarray]:
        """Get states of nodes filtered by type.

        Args:
            node_type: "edge" or "center".

        Returns:
            Dict mapping node_id to state vector.
        """
        with self._lock:
            return {
                nid: node.to_state_vector()
                for nid, node in self._nodes.items()
                if node.registered and node.node_type == node_type
            }

    def get_nodes_by_service(self, service_type: str) -> Dict[str, np.ndarray]:
        """Get states of nodes that support a specific service type.

        Args:
            service_type: Service type string (e.g., "inference", "training").

        Returns:
            Dict mapping node_id to state vector.
        """
        with self._lock:
            return {
                nid: node.to_state_vector()
                for nid, node in self._nodes.items()
                if node.registered and service_type in node.service_types
            }

    def get_history(self, node_id: str) -> List[Dict]:
        """Get the state update history for a node."""
        with self._lock:
            return list(self._history.get(node_id, []))

    def get_all_node_ids(self) -> List[str]:
        """Get list of all registered node IDs."""
        with self._lock:
            return [
                nid for nid, node in self._nodes.items() if node.registered
            ]

    def simulate_task_arrival(
        self, node_id: str, gpu_increase: float = 0.1, queue_increase: int = 1
    ) -> None:
        """Simulate the effect of a task arriving at a computing node.

        Increases GPU utilization and task queue length.

        Args:
            node_id: Target node.
            gpu_increase: Amount to increase GPU utilization.
            queue_increase: Number of tasks to add to queue.
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' is not registered")
            node = self._nodes[node_id]
            node.gpu_utilization = min(1.0, node.gpu_utilization + gpu_increase)
            node.task_queue_length += queue_increase
            node.cpu_load = min(1.0, node.cpu_load + gpu_increase * 0.5)
            node.last_update = time.time()
            self._history[node_id].append({
                "timestamp": node.last_update,
                "state": node.to_state_vector().tolist(),
                "event": "task_arrival",
            })

    def simulate_task_completion(
        self, node_id: str, gpu_decrease: float = 0.1, queue_decrease: int = 1
    ) -> None:
        """Simulate the effect of a task completing at a computing node.

        Decreases GPU utilization and task queue length.

        Args:
            node_id: Target node.
            gpu_decrease: Amount to decrease GPU utilization.
            queue_decrease: Number of tasks to remove from queue.
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node '{node_id}' is not registered")
            node = self._nodes[node_id]
            node.gpu_utilization = max(0.0, node.gpu_utilization - gpu_decrease)
            node.task_queue_length = max(
                0.0, node.task_queue_length - queue_decrease
            )
            node.cpu_load = max(0.0, node.cpu_load - gpu_decrease * 0.5)
            node.last_update = time.time()
            self._history[node_id].append({
                "timestamp": node.last_update,
                "state": node.to_state_vector().tolist(),
                "event": "task_completion",
            })

    def __repr__(self) -> str:
        registered = sum(1 for n in self._nodes.values() if n.registered)
        return f"CPR(registered_nodes={registered})"
