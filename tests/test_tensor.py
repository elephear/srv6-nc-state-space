"""Tests for the JointStateTensor module."""

import numpy as np
import pytest

from modules.state_tensor import (
    COMP_CPU,
    COMP_GPU,
    COMP_REACHABILITY,
    COMP_TASK_QUEUE,
    JointStateTensor,
    NET_BANDWIDTH,
    NET_DELAY,
    NET_UTILIZATION,
    STATE_DIM,
)


@pytest.fixture
def simple_tensor():
    """Create a simple tensor with 4 nodes, 2 of which are compute nodes."""
    node_ids = ["R1", "R2", "C1", "C2"]
    compute_ids = ["C1", "C2"]
    return JointStateTensor(node_ids, compute_ids)


class TestTensorInitialization:
    def test_shape(self, simple_tensor):
        assert simple_tensor.shape == (4, 4, STATE_DIM)

    def test_initial_zeros(self, simple_tensor):
        assert np.all(simple_tensor.tensor == 0.0)

    def test_node_count(self, simple_tensor):
        assert simple_tensor.n == 4

    def test_compute_node_ids(self, simple_tensor):
        assert simple_tensor.compute_node_ids == ["C1", "C2"]

    def test_invalid_compute_node(self):
        with pytest.raises(ValueError, match="not found"):
            JointStateTensor(["R1", "R2"], ["C1"])


class TestLinkStateOperations:
    def test_update_link_state(self, simple_tensor):
        simple_tensor.update_link_state(
            "R1", "R2", bandwidth=0.8, delay=0.1, utilization=0.3
        )
        state = simple_tensor.get_link_state("R1", "R2")
        assert state[NET_BANDWIDTH] == pytest.approx(0.8)
        assert state[NET_DELAY] == pytest.approx(0.1)
        assert state[NET_UTILIZATION] == pytest.approx(0.3)

    def test_set_link_state_vector(self, simple_tensor):
        vec = np.array([0.9, 0.05, 0.01, 0.2, 0.03])
        simple_tensor.set_link_state("R1", "C1", vec)
        result = simple_tensor.get_link_state("R1", "C1")
        np.testing.assert_array_almost_equal(result, vec)

    def test_set_link_state_wrong_dim(self, simple_tensor):
        with pytest.raises(ValueError, match="5 dimensions"):
            simple_tensor.set_link_state("R1", "R2", np.array([1, 2, 3]))

    def test_invalid_node_id(self, simple_tensor):
        with pytest.raises(KeyError, match="not found"):
            simple_tensor.update_link_state("X1", "R2", bandwidth=0.5)

    def test_partial_update(self, simple_tensor):
        simple_tensor.update_link_state("R1", "R2", bandwidth=0.5)
        simple_tensor.update_link_state("R1", "R2", delay=0.2)
        state = simple_tensor.get_link_state("R1", "R2")
        assert state[NET_BANDWIDTH] == pytest.approx(0.5)
        assert state[NET_DELAY] == pytest.approx(0.2)


class TestComputeStateOperations:
    def test_update_compute_state(self, simple_tensor):
        simple_tensor.update_compute_state(
            "C1", gpu=0.6, cpu=0.4, task_queue=0.3, reachability=0.95
        )
        state = simple_tensor.get_compute_state("C1")
        assert state[COMP_GPU] == pytest.approx(0.6)
        assert state[COMP_CPU] == pytest.approx(0.4)
        assert state[COMP_TASK_QUEUE] == pytest.approx(0.3)
        assert state[COMP_REACHABILITY] == pytest.approx(0.95)

    def test_set_compute_state_vector(self, simple_tensor):
        vec = np.array([0.5, 0.3, 0.2, 0.1, 1.0])
        simple_tensor.set_compute_state("C2", vec)
        result = simple_tensor.get_compute_state("C2")
        np.testing.assert_array_almost_equal(result, vec)

    def test_non_compute_node_raises(self, simple_tensor):
        with pytest.raises(ValueError, match="not a registered computing node"):
            simple_tensor.update_compute_state("R1", gpu=0.5)

    def test_diagonal_storage(self, simple_tensor):
        """Compute state should be stored on the diagonal of the tensor."""
        vec = np.array([0.5, 0.3, 0.2, 0.1, 1.0])
        simple_tensor.set_compute_state("C1", vec)
        # C1 is at index 2
        np.testing.assert_array_almost_equal(
            simple_tensor.tensor[2, 2, :], vec
        )

    def test_get_compute_states_dict(self, simple_tensor):
        simple_tensor.set_compute_state("C1", np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        simple_tensor.set_compute_state("C2", np.array([0.5, 0.4, 0.3, 0.2, 0.1]))
        states = simple_tensor.get_compute_states_dict()
        assert len(states) == 2
        assert "C1" in states
        assert "C2" in states


class TestTensorOperations:
    def test_adjacency_matrix(self, simple_tensor):
        simple_tensor.update_link_state("R1", "R2", utilization=0.5)
        simple_tensor.update_link_state("R2", "R1", utilization=0.3)
        matrix = simple_tensor.get_adjacency_matrix(NET_UTILIZATION)
        assert matrix.shape == (4, 4)
        assert matrix[0, 1] == pytest.approx(0.5)
        assert matrix[1, 0] == pytest.approx(0.3)

    def test_snapshot_is_copy(self, simple_tensor):
        simple_tensor.update_link_state("R1", "R2", bandwidth=0.7)
        snapshot = simple_tensor.get_snapshot()
        snapshot[0, 1, NET_BANDWIDTH] = 999.0
        # Original should not be affected
        assert simple_tensor.tensor[0, 1, NET_BANDWIDTH] == pytest.approx(0.7)

    def test_get_link_state_returns_copy(self, simple_tensor):
        simple_tensor.update_link_state("R1", "R2", bandwidth=0.7)
        state = simple_tensor.get_link_state("R1", "R2")
        state[NET_BANDWIDTH] = 999.0
        assert simple_tensor.tensor[0, 1, NET_BANDWIDTH] == pytest.approx(0.7)

    def test_repr(self, simple_tensor):
        r = repr(simple_tensor)
        assert "nodes=4" in r
        assert "compute_nodes=2" in r
