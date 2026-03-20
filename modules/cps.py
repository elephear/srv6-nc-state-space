"""
Computing Power Service (CPS) Node Module.

Simulates edge and center computing nodes. In the Mininet environment,
each CPS runs as a Python process on a host, listening for task requests
on UDP port 5000. On receiving a task, it:
  1. Parses the task requirements from the packet header.
  2. Calls CPR.update_cps() to reflect resource consumption.
  3. Simulates processing delay.
  4. Returns a response packet.

For standalone simulation (without Mininet), this module provides a
CPSSimulator class that models CPS behavior programmatically.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TaskRequest:
    """Represents an incoming task request."""

    task_id: str
    task_type: str  # "inference", "training", etc.
    compute_req: float  # Required compute (normalized)
    latency_req: float  # Maximum latency in ms
    data_size: float  # Data size in MB
    source_node: str  # Originating user node
    arrival_time: float = 0.0


@dataclass
class TaskResult:
    """Result of a completed task."""

    task_id: str
    node_id: str
    start_time: float
    end_time: float
    processing_time: float
    success: bool
    path: List[str] = field(default_factory=list)


class CPSSimulator:
    """Simulates a Computing Power Service node.

    Models the behavior of a CPS node including resource consumption,
    task queuing, and state reporting to CPR.
    """

    def __init__(
        self,
        node_id: str,
        node_type: str = "edge",
        max_gpu_capacity: float = 1.0,
        max_cpu_capacity: float = 1.0,
        max_queue_length: int = 100,
        base_energy: float = 0.1,
        processing_rate: float = 0.1,
    ):
        """Initialize CPS simulator.

        Args:
            node_id: Unique identifier.
            node_type: "edge" or "center".
            max_gpu_capacity: Maximum GPU capacity (normalized).
            max_cpu_capacity: Maximum CPU capacity (normalized).
            max_queue_length: Maximum task queue size.
            base_energy: Base energy weight per unit task.
            processing_rate: Tasks processed per second (simulated).
        """
        self.node_id = node_id
        self.node_type = node_type
        self.max_gpu_capacity = max_gpu_capacity
        self.max_cpu_capacity = max_cpu_capacity
        self.max_queue_length = max_queue_length
        self.base_energy = base_energy
        self.processing_rate = processing_rate

        # Current state
        self.gpu_utilization = 0.0
        self.cpu_load = 0.0
        self.task_queue: List[TaskRequest] = []
        self.reachability = 1.0

        # Tracking
        self.completed_tasks: List[TaskResult] = []
        self.total_tasks_received = 0

    @property
    def task_queue_length(self) -> int:
        """Current number of tasks in queue."""
        return len(self.task_queue)

    @property
    def normalized_queue_length(self) -> float:
        """Queue length normalized by max capacity."""
        return len(self.task_queue) / self.max_queue_length

    @property
    def energy_weight(self) -> float:
        """Dynamic energy weight based on current utilization."""
        # Energy increases with utilization (dynamic power)
        return self.base_energy * (1.0 + self.gpu_utilization)

    def get_state_vector(self) -> np.ndarray:
        """Get current 5D state vector [G, L, Q^task, P, R]."""
        return np.array([
            self.gpu_utilization,
            self.cpu_load,
            self.normalized_queue_length,
            self.energy_weight,
            self.reachability,
        ])

    def receive_task(self, task: TaskRequest) -> bool:
        """Receive and enqueue a task.

        Args:
            task: The incoming task request.

        Returns:
            True if task was accepted, False if queue is full.
        """
        if len(self.task_queue) >= self.max_queue_length:
            return False

        task.arrival_time = time.time()
        self.task_queue.append(task)
        self.total_tasks_received += 1

        # Update resource consumption
        gpu_per_task = task.compute_req / self.max_gpu_capacity
        self.gpu_utilization = min(
            1.0, self.gpu_utilization + gpu_per_task * 0.2
        )
        self.cpu_load = min(1.0, self.cpu_load + gpu_per_task * 0.1)

        return True

    def process_next_task(self) -> Optional[TaskResult]:
        """Process the next task in queue.

        Returns:
            TaskResult if a task was processed, None if queue is empty.
        """
        if not self.task_queue:
            return None

        task = self.task_queue.pop(0)
        start_time = time.time()

        # Simulate processing time based on compute requirement
        processing_time = task.compute_req / self.processing_rate
        # Add noise
        processing_time *= 1.0 + random.uniform(-0.1, 0.1)

        end_time = start_time + processing_time
        success = processing_time * 1000 <= task.latency_req  # Check SLA

        # Release resources
        gpu_per_task = task.compute_req / self.max_gpu_capacity
        self.gpu_utilization = max(
            0.0, self.gpu_utilization - gpu_per_task * 0.2
        )
        self.cpu_load = max(0.0, self.cpu_load - gpu_per_task * 0.1)

        result = TaskResult(
            task_id=task.task_id,
            node_id=self.node_id,
            start_time=start_time,
            end_time=end_time,
            processing_time=processing_time,
            success=success,
        )
        self.completed_tasks.append(result)
        return result

    def set_reachability(self, probe_result: int, alpha: float = 0.2) -> None:
        """Update reachability using EWMA.

        R(t) = alpha * probe(t) + (1 - alpha) * R(t-1)
        """
        self.reachability = alpha * probe_result + (1 - alpha) * self.reachability

    def simulate_failure(self) -> None:
        """Simulate a node failure."""
        self.reachability = 0.0

    def simulate_recovery(self) -> None:
        """Simulate node recovery."""
        self.reachability = 1.0

    def __repr__(self) -> str:
        return (
            f"CPS({self.node_id}, type={self.node_type}, "
            f"gpu={self.gpu_utilization:.2f}, queue={self.task_queue_length})"
        )
