"""
Joint State Tensor S ∈ R^{|V|x|V|x5}

Unified representation of network link states and computing node states.

For each directed edge e(i,j) in E:
    S[i,j] = [B_ij, D_ij, J_ij, U_ij, Q_ij^pkt]
    (bandwidth, delay, jitter, utilization, packet queue length)

For each computing node n in V_c:
    S[n,n] = [G_n, L_n, Q_n^task, P_n, R_n]
    (GPU intensity, CPU load, task queue length, energy weight, reachability)

All other entries are zero vectors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# Network state dimension indices
NET_BANDWIDTH = 0
NET_DELAY = 1
NET_JITTER = 2
NET_UTILIZATION = 3
NET_QUEUE = 4

# Computing state dimension indices (on diagonal)
COMP_GPU = 0
COMP_CPU = 1
COMP_TASK_QUEUE = 2
COMP_ENERGY = 3
COMP_REACHABILITY = 4

STATE_DIM = 5


class JointStateTensor:
    """Joint network-computing state tensor S ∈ R^{|V|x|V|x5}.

    Maintains a unified representation of network link states and computing
    node resource states in a single 3D tensor structure.
    """

    def __init__(self, node_ids: List[str], compute_node_ids: List[str]):
        """Initialize the joint state tensor.

        Args:
            node_ids: List of all node identifiers in the network.
            compute_node_ids: List of computing node identifiers (subset of node_ids).
        """
        self.node_ids = list(node_ids)
        self.compute_node_ids = list(compute_node_ids)
        self.n = len(self.node_ids)
        self._id_to_idx: Dict[str, int] = {
            nid: i for i, nid in enumerate(self.node_ids)
        }

        # Validate compute nodes are a subset of all nodes
        for cn in self.compute_node_ids:
            if cn not in self._id_to_idx:
                raise ValueError(
                    f"Computing node '{cn}' not found in node_ids"
                )

        # Initialize tensor with zeros
        self.tensor = np.zeros((self.n, self.n, STATE_DIM), dtype=np.float64)

        # Track which edges exist
        self._edges: set = set()

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return the shape of the tensor."""
        return self.tensor.shape

    def _get_idx(self, node_id: str) -> int:
        """Get the index of a node by its ID."""
        if node_id not in self._id_to_idx:
            raise KeyError(f"Node '{node_id}' not found in tensor")
        return self._id_to_idx[node_id]

    # --- Edge (link) state operations ---

    def add_edge(self, src: str, dst: str) -> None:
        """Register an edge in the topology."""
        self._edges.add((src, dst))

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
        """Update the state vector for a network link.

        Args:
            src: Source node ID.
            dst: Destination node ID.
            bandwidth: Available bandwidth (normalized to [0,1]).
            delay: Transmission delay (normalized to [0,1]).
            jitter: Delay variance (normalized to [0,1]).
            utilization: Link utilization ratio [0,1].
            queue_length: Packet queue length (normalized to [0,1]).
        """
        i = self._get_idx(src)
        j = self._get_idx(dst)

        if bandwidth is not None:
            self.tensor[i, j, NET_BANDWIDTH] = bandwidth
        if delay is not None:
            self.tensor[i, j, NET_DELAY] = delay
        if jitter is not None:
            self.tensor[i, j, NET_JITTER] = jitter
        if utilization is not None:
            self.tensor[i, j, NET_UTILIZATION] = utilization
        if queue_length is not None:
            self.tensor[i, j, NET_QUEUE] = queue_length

    def set_link_state(
        self, src: str, dst: str, state_vector: np.ndarray
    ) -> None:
        """Set the full 5D state vector for a link.

        Args:
            src: Source node ID.
            dst: Destination node ID.
            state_vector: 5-element array [B, D, J, U, Q].
        """
        if len(state_vector) != STATE_DIM:
            raise ValueError(
                f"State vector must have {STATE_DIM} dimensions, "
                f"got {len(state_vector)}"
            )
        i = self._get_idx(src)
        j = self._get_idx(dst)
        self.tensor[i, j, :] = state_vector

    def get_link_state(self, src: str, dst: str) -> np.ndarray:
        """Get the state vector for a link.

        Returns:
            5-element array [B, D, J, U, Q].
        """
        i = self._get_idx(src)
        j = self._get_idx(dst)
        return self.tensor[i, j, :].copy()

    # --- Computing node state operations ---

    def update_compute_state(
        self,
        node_id: str,
        gpu: Optional[float] = None,
        cpu: Optional[float] = None,
        task_queue: Optional[float] = None,
        energy: Optional[float] = None,
        reachability: Optional[float] = None,
    ) -> None:
        """Update the state vector for a computing node.

        Args:
            node_id: Computing node ID.
            gpu: GPU utilization (normalized to [0,1]).
            cpu: CPU load (normalized to [0,1]).
            task_queue: Task queue length (normalized to [0,1]).
            energy: Energy weight per unit task (normalized to [0,1]).
            reachability: Dynamic reachability probability [0,1].
        """
        if node_id not in self.compute_node_ids:
            raise ValueError(
                f"Node '{node_id}' is not a registered computing node"
            )
        n = self._get_idx(node_id)

        if gpu is not None:
            self.tensor[n, n, COMP_GPU] = gpu
        if cpu is not None:
            self.tensor[n, n, COMP_CPU] = cpu
        if task_queue is not None:
            self.tensor[n, n, COMP_TASK_QUEUE] = task_queue
        if energy is not None:
            self.tensor[n, n, COMP_ENERGY] = energy
        if reachability is not None:
            self.tensor[n, n, COMP_REACHABILITY] = reachability

    def set_compute_state(
        self, node_id: str, state_vector: np.ndarray
    ) -> None:
        """Set the full 5D state vector for a computing node.

        Args:
            node_id: Computing node ID.
            state_vector: 5-element array [G, L, Q^task, P, R].
        """
        if len(state_vector) != STATE_DIM:
            raise ValueError(
                f"State vector must have {STATE_DIM} dimensions, "
                f"got {len(state_vector)}"
            )
        if node_id not in self.compute_node_ids:
            raise ValueError(
                f"Node '{node_id}' is not a registered computing node"
            )
        n = self._get_idx(node_id)
        self.tensor[n, n, :] = state_vector

    def get_compute_state(self, node_id: str) -> np.ndarray:
        """Get the state vector for a computing node.

        Returns:
            5-element array [G, L, Q^task, P, R].
        """
        if node_id not in self.compute_node_ids:
            raise ValueError(
                f"Node '{node_id}' is not a registered computing node"
            )
        n = self._get_idx(node_id)
        return self.tensor[n, n, :].copy()

    # --- Tensor-level operations ---

    def get_adjacency_matrix(self, dim: int) -> np.ndarray:
        """Extract a 2D adjacency matrix for a specific state dimension.

        Args:
            dim: State dimension index (0-4).

        Returns:
            |V|x|V| matrix for the specified dimension.
        """
        if dim < 0 or dim >= STATE_DIM:
            raise ValueError(f"Dimension must be 0-{STATE_DIM - 1}, got {dim}")
        return self.tensor[:, :, dim].copy()

    def get_network_utilization_matrix(self) -> np.ndarray:
        """Get the link utilization adjacency matrix."""
        return self.get_adjacency_matrix(NET_UTILIZATION)

    def get_snapshot(self) -> np.ndarray:
        """Get a full copy of the current tensor state."""
        return self.tensor.copy()

    def get_compute_states_dict(self) -> Dict[str, np.ndarray]:
        """Get all computing node states as a dictionary.

        Returns:
            Dict mapping node_id to 5D state vector.
        """
        result = {}
        for nid in self.compute_node_ids:
            result[nid] = self.get_compute_state(nid)
        return result

    def get_edges(self) -> set:
        """Return the set of registered edges."""
        return self._edges.copy()

    def __repr__(self) -> str:
        return (
            f"JointStateTensor(nodes={self.n}, "
            f"compute_nodes={len(self.compute_node_ids)}, "
            f"shape={self.shape})"
        )
