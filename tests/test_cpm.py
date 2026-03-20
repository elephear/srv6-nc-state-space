"""Tests for the CPM (Controller Policy Module)."""

import numpy as np
import pytest
import networkx as nx

from modules.cpm import CPM
from modules.cpr import CPR
from modules.state_tensor import JointStateTensor, NET_UTILIZATION, NET_DELAY


@pytest.fixture
def simple_network():
    """Create a simple linear network: U1 - R1 - R2 - R3 - C1."""
    node_ids = ["U1", "R1", "R2", "R3", "C1"]
    compute_ids = ["C1"]

    graph = nx.DiGraph()
    graph.add_edges_from([
        ("U1", "R1"), ("R1", "U1"),
        ("R1", "R2"), ("R2", "R1"),
        ("R2", "R3"), ("R3", "R2"),
        ("R3", "C1"), ("C1", "R3"),
    ])

    tensor = JointStateTensor(node_ids, compute_ids)
    # Initialize link states
    for src, dst in graph.edges():
        tensor.set_link_state(src, dst, np.array([0.9, 0.1, 0.02, 0.15, 0.05]))

    cpr = CPR()
    cpr.register_node(
        "C1", node_type="center",
        service_types=["inference"],
        initial_state=np.array([0.3, 0.2, 0.1, 0.1, 1.0]),
    )

    cpm = CPM(tensor=tensor, cpr=cpr, graph=graph)
    return cpm, tensor, cpr, graph


@pytest.fixture
def branching_network():
    """Create a network with two paths: U1-R1-R2-C1 and U1-R1-R3-C1."""
    node_ids = ["U1", "R1", "R2", "R3", "C1", "C2"]
    compute_ids = ["C1", "C2"]

    graph = nx.DiGraph()
    graph.add_edges_from([
        ("U1", "R1"), ("R1", "U1"),
        ("R1", "R2"), ("R2", "R1"),
        ("R1", "R3"), ("R3", "R1"),
        ("R2", "C1"), ("C1", "R2"),
        ("R3", "C1"), ("C1", "R3"),
        ("R3", "C2"), ("C2", "R3"),
    ])

    tensor = JointStateTensor(node_ids, compute_ids)
    for src, dst in graph.edges():
        tensor.set_link_state(src, dst, np.array([0.9, 0.1, 0.02, 0.15, 0.05]))

    cpr = CPR()
    cpr.register_node(
        "C1", node_type="center",
        service_types=["inference"],
        initial_state=np.array([0.3, 0.2, 0.1, 0.1, 1.0]),
    )
    cpr.register_node(
        "C2", node_type="center",
        service_types=["inference", "training"],
        initial_state=np.array([0.2, 0.15, 0.05, 0.08, 1.0]),
    )

    cpm = CPM(tensor=tensor, cpr=cpr, graph=graph)
    return cpm, tensor, cpr, graph


class TestCPMPathRequest:
    def test_basic_path(self, simple_network):
        cpm, tensor, cpr, graph = simple_network
        result = cpm.path_request(src="U1", dst="C1")
        assert result is not None
        target, sid_list, cost = result
        assert target == "C1"
        assert len(sid_list) > 0
        assert cost > 0

    def test_path_includes_correct_nodes(self, simple_network):
        cpm, tensor, cpr, graph = simple_network
        result = cpm.path_request(src="U1", dst="C1")
        target, sid_list, cost = result
        full_path = ["U1"] + sid_list
        assert full_path[0] == "U1"
        assert full_path[-1] == "C1"

    def test_no_path_returns_none(self, simple_network):
        cpm, tensor, cpr, graph = simple_network
        # Add isolated node
        tensor_new = JointStateTensor(
            ["U1", "R1", "R2", "R3", "C1", "X1"],
            ["C1", "X1"],
        )
        cpr_new = CPR()
        cpr_new.register_node("C1", initial_state=np.array([0.3, 0.2, 0.1, 0.1, 1.0]))
        cpr_new.register_node("X1", initial_state=np.array([0.1, 0.1, 0.0, 0.1, 1.0]))
        # X1 is not connected
        cpm_new = CPM(tensor=tensor_new, cpr=cpr_new, graph=graph)
        result = cpm_new.path_request(src="U1", dst="X1")
        assert result is None


class TestCPMRerouting:
    def test_reroute_on_congestion(self, branching_network):
        cpm, tensor, cpr, graph = branching_network

        # Initial: both paths have equal cost, get initial path
        result1 = cpm.path_request(src="U1", dst="C1")
        assert result1 is not None
        path1 = ["U1"] + result1[1]

        # Congest the R1->R2 link
        tensor.update_link_state("R1", "R2", utilization=0.95, delay=0.8)

        result2 = cpm.path_request(src="U1", dst="C1")
        assert result2 is not None
        path2 = ["U1"] + result2[1]

        # Path should now prefer R3 route
        assert "R3" in path2

    def test_compute_node_selection(self, branching_network):
        cpm, tensor, cpr, graph = branching_network

        # Make C1 heavily loaded, C2 lightly loaded
        cpr.update_cps("C1", gpu=0.9, task_queue=0.8)
        cpr.update_cps("C2", gpu=0.1, task_queue=0.05)

        # Request without specifying destination
        result = cpm.path_request(src="U1")
        assert result is not None
        target = result[0]
        # Should prefer C2 (lower load)
        assert target == "C2"

    def test_reachability_filter(self, branching_network):
        cpm, tensor, cpr, graph = branching_network

        # Make C1 unreachable
        cpr.update_cps("C1", new_state=np.array([0.3, 0.2, 0.1, 0.1, 0.1]))

        # Request without dst - should only consider C2
        result = cpm.path_request(src="U1")
        assert result is not None
        assert result[0] == "C2"


class TestCPMLinkCosts:
    def test_link_cost_increases_with_utilization(self, simple_network):
        cpm, tensor, cpr, graph = simple_network

        tensor.update_link_state("R1", "R2", utilization=0.1)
        cost_low = cpm._compute_link_cost("R1", "R2")

        tensor.update_link_state("R1", "R2", utilization=0.9)
        cost_high = cpm._compute_link_cost("R1", "R2")

        assert cost_high > cost_low

    def test_link_cost_increases_with_delay(self, simple_network):
        cpm, tensor, cpr, graph = simple_network

        tensor.update_link_state("R1", "R2", delay=0.05)
        cost_low = cpm._compute_link_cost("R1", "R2")

        tensor.update_link_state("R1", "R2", delay=0.8)
        cost_high = cpm._compute_link_cost("R1", "R2")

        assert cost_high > cost_low

    def test_node_cost_increases_with_gpu_load(self, branching_network):
        cpm, tensor, cpr, graph = branching_network

        cpr.update_cps("C1", gpu=0.1)
        cpm.sync_compute_state()
        cost_low = cpm._compute_node_cost("C1")

        cpr.update_cps("C1", gpu=0.9)
        cpm.sync_compute_state()
        cost_high = cpm._compute_node_cost("C1")

        assert cost_high > cost_low


class TestCPMSync:
    def test_sync_compute_state(self, simple_network):
        cpm, tensor, cpr, graph = simple_network
        cpr.update_cps("C1", gpu=0.8, cpu=0.6)
        cpm.sync_compute_state()
        state = tensor.get_compute_state("C1")
        assert state[0] == pytest.approx(0.8)
        assert state[1] == pytest.approx(0.6)

    def test_get_all_link_costs(self, simple_network):
        cpm, tensor, cpr, graph = simple_network
        costs = cpm.get_all_link_costs()
        assert len(costs) == graph.number_of_edges()
        for cost in costs.values():
            assert cost >= 0

    def test_get_all_node_costs(self, branching_network):
        cpm, tensor, cpr, graph = branching_network
        cpm.sync_compute_state()
        costs = cpm.get_all_node_costs()
        assert "C1" in costs
        assert "C2" in costs
