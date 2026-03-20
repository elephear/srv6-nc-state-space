"""Tests for the CPR (Computing Power Registry) module."""

import numpy as np
import pytest

from modules.cpr import CPR


@pytest.fixture
def cpr():
    """Create a CPR with some registered nodes."""
    registry = CPR(reachability_alpha=0.2)
    registry.register_node(
        "E1", node_type="edge",
        service_types=["inference", "container"],
        initial_state=np.array([0.2, 0.15, 0.0, 0.1, 1.0]),
    )
    registry.register_node(
        "E2", node_type="edge",
        service_types=["inference"],
        initial_state=np.array([0.3, 0.2, 0.0, 0.12, 1.0]),
    )
    registry.register_node(
        "C1", node_type="center",
        service_types=["inference", "training", "vm"],
        initial_state=np.array([0.4, 0.3, 0.1, 0.15, 1.0]),
    )
    return registry


class TestCPRRegistration:
    def test_register_node(self, cpr):
        ids = cpr.get_all_node_ids()
        assert "E1" in ids
        assert "E2" in ids
        assert "C1" in ids

    def test_initial_state(self, cpr):
        state = cpr.get_node_state("E1")
        assert state[0] == pytest.approx(0.2)  # GPU
        assert state[4] == pytest.approx(1.0)  # Reachability

    def test_unregister_node(self, cpr):
        cpr.unregister_node("E1")
        states = cpr.get_cps()
        assert "E1" not in states

    def test_get_unregistered_raises(self, cpr):
        with pytest.raises(KeyError):
            cpr.get_node_state("NONEXISTENT")


class TestCPRStateUpdate:
    def test_update_full_state(self, cpr):
        new_state = np.array([0.8, 0.6, 0.5, 0.2, 0.9])
        cpr.update_cps("E1", new_state=new_state)
        result = cpr.get_node_state("E1")
        np.testing.assert_array_almost_equal(result, new_state)

    def test_update_individual_fields(self, cpr):
        cpr.update_cps("E1", gpu=0.7, cpu=0.5)
        state = cpr.get_node_state("E1")
        assert state[0] == pytest.approx(0.7)
        assert state[1] == pytest.approx(0.5)
        # Other fields unchanged
        assert state[2] == pytest.approx(0.0)  # task queue

    def test_gpu_clipping(self, cpr):
        cpr.update_cps("E1", gpu=1.5)
        state = cpr.get_node_state("E1")
        assert state[0] == pytest.approx(1.0)

    def test_reachability_ewma(self, cpr):
        # Initial R=1.0, alpha=0.2
        # After probe=0: R = 0.2*0 + 0.8*1.0 = 0.8
        cpr.update_cps("E1", probe_result=0)
        state = cpr.get_node_state("E1")
        assert state[4] == pytest.approx(0.8)

        # After another probe=0: R = 0.2*0 + 0.8*0.8 = 0.64
        cpr.update_cps("E1", probe_result=0)
        state = cpr.get_node_state("E1")
        assert state[4] == pytest.approx(0.64)

    def test_update_nonexistent_raises(self, cpr):
        with pytest.raises(KeyError):
            cpr.update_cps("NONEXISTENT", gpu=0.5)


class TestCPRFiltering:
    def test_get_nodes_by_type_edge(self, cpr):
        edges = cpr.get_nodes_by_type("edge")
        assert "E1" in edges
        assert "E2" in edges
        assert "C1" not in edges

    def test_get_nodes_by_type_center(self, cpr):
        centers = cpr.get_nodes_by_type("center")
        assert "C1" in centers
        assert "E1" not in centers

    def test_get_nodes_by_service(self, cpr):
        inference = cpr.get_nodes_by_service("inference")
        assert len(inference) == 3  # E1, E2, C1

        training = cpr.get_nodes_by_service("training")
        assert len(training) == 1  # C1 only

        container = cpr.get_nodes_by_service("container")
        assert len(container) == 1  # E1 only


class TestCPRSimulation:
    def test_task_arrival(self, cpr):
        initial_gpu = cpr.get_node_state("E1")[0]
        cpr.simulate_task_arrival("E1", gpu_increase=0.2, queue_increase=1)
        state = cpr.get_node_state("E1")
        assert state[0] == pytest.approx(initial_gpu + 0.2)
        assert state[2] >= 1.0  # Queue increased

    def test_task_completion(self, cpr):
        cpr.simulate_task_arrival("E1", gpu_increase=0.3, queue_increase=2)
        state_after_arrival = cpr.get_node_state("E1")

        cpr.simulate_task_completion("E1", gpu_decrease=0.1, queue_decrease=1)
        state_after_completion = cpr.get_node_state("E1")

        assert state_after_completion[0] < state_after_arrival[0]

    def test_gpu_capped_at_1(self, cpr):
        for _ in range(20):
            cpr.simulate_task_arrival("E1", gpu_increase=0.2)
        state = cpr.get_node_state("E1")
        assert state[0] <= 1.0

    def test_gpu_floored_at_0(self, cpr):
        for _ in range(20):
            cpr.simulate_task_completion("E1", gpu_decrease=0.2)
        state = cpr.get_node_state("E1")
        assert state[0] >= 0.0

    def test_history_tracking(self, cpr):
        cpr.simulate_task_arrival("E1")
        cpr.simulate_task_arrival("E1")
        history = cpr.get_history("E1")
        # Initial registration + 2 arrivals
        assert len(history) >= 2
