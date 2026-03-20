"""
Visualization and Plotting for Experiment Results.

Generates paper-ready figures:
  - Figure 6: Link utilization change and S update curve (Experiment 1)
  - Figure 7: Path comparison before/after congestion (Experiment 2)
  - Topology visualization (Figure 5)
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a clean style suitable for academic papers
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.linewidth": 2,
    "grid.alpha": 0.3,
})


def plot_exp1_utilization(
    results_path: str = "results/exp1/exp1_results.json",
    output_path: str = "results/exp1/fig6_utilization_update.png",
):
    """Plot Figure 6: Link utilization change and tensor S update.

    Shows the actual link utilization change event and how the tensor S
    captures the change with minimal delay.
    """
    with open(results_path) as f:
        results = json.load(f)

    timestamps = results["timestamps"]
    util_before = results["utilization_before_update"]
    util_after = results["utilization_after_update"]
    event_time = results["parameters"]["event_time"]

    fig, ax = plt.subplots()

    # Actual utilization (step function at event time)
    actual_util = []
    for t in timestamps:
        if t < event_time:
            actual_util.append(0.10)
        else:
            actual_util.append(0.70)

    ax.plot(
        timestamps, actual_util,
        "r--", label="Actual link utilization", linewidth=2, alpha=0.8,
    )
    ax.plot(
        timestamps, util_after,
        "b-o", label="$\\mathcal{S}_{R12,R22}$ (utilization in tensor)",
        markersize=3, linewidth=2,
    )

    # Mark the event
    ax.axvline(
        x=event_time, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
    )
    ax.annotate(
        "Traffic injection\nevent",
        xy=(event_time, 0.40),
        xytext=(event_time + 0.5, 0.45),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    # Highlight detection delay
    detection = results["verification"]["detection_delay_s"]
    if detection is not None:
        detect_t = event_time + detection
        ax.axvline(
            x=detect_t, color="green", linestyle="--",
            linewidth=1.5, alpha=0.5,
        )
        ax.annotate(
            f"Detected\n({detection:.2f}s delay)",
            xy=(detect_t, 0.65),
            xytext=(detect_t + 0.3, 0.55),
            fontsize=9,
            color="green",
            arrowprops=dict(arrowstyle="->", color="green"),
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Link Utilization ($U_{R12,R22}$)")
    ax.set_title("Link Utilization Change and $\\mathcal{S}$ Update (R12-R22)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Figure 6 saved to {output_path}")


def plot_exp2_path_comparison(
    results_path: str = "results/exp2/exp2_results.json",
    output_path: str = "results/exp2/fig7_path_comparison.png",
):
    """Plot Figure 7: Path comparison before and after congestion.

    Shows the topology with two paths highlighted:
    - Initial path (blue)
    - Rerouted path after congestion (green)
    - Congested link (red)
    """
    with open(results_path) as f:
        results = json.load(f)

    initial_path = results["initial_path"]["path"]
    updated_path = results["updated_path"]["path"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, path, title, color in [
        (ax1, initial_path, "Before Congestion (Initial Path)", "blue"),
        (ax2, updated_path, "After Congestion (Rerouted Path)", "green"),
    ]:
        _draw_topology(ax, path, color, title)

    fig.suptitle(
        "Path Decision Impact: Rerouting on Link Congestion",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Figure 7 saved to {output_path}")


def _draw_topology(ax, highlight_path, path_color, title):
    """Draw the 4x4 grid topology with a highlighted path.

    Args:
        ax: Matplotlib axes.
        highlight_path: List of node IDs forming the path to highlight.
        path_color: Color for the highlighted path.
        title: Subplot title.
    """
    # Node positions
    positions = {}

    # Core routers: 4x4 grid
    for row in range(1, 5):
        for col in range(1, 5):
            rid = f"R{row}{col}"
            positions[rid] = (col, 5 - row)

    # Edge compute nodes (above rows 1-2)
    edge_positions = {
        "E1": (1, 5.5), "E2": (2, 5.5), "E3": (3, 5.5), "E4": (4, 5.5),
        "E5": (1, 4.5), "E6": (2, 4.5), "E7": (3, 4.5), "E8": (4, 4.5),
    }
    # Adjust: E5-E8 slightly offset
    edge_positions["E5"] = (0.3, 3.5)
    edge_positions["E6"] = (0.3, 3.0)
    edge_positions["E7"] = (4.7, 3.5)
    edge_positions["E8"] = (4.7, 3.0)
    # E1-E4 above row 1
    edge_positions["E1"] = (0.3, 4.5)
    edge_positions["E2"] = (0.3, 4.0)
    edge_positions["E3"] = (4.7, 4.5)
    edge_positions["E4"] = (4.7, 4.0)
    positions.update(edge_positions)

    # Center compute nodes (below row 3-4)
    positions["C1"] = (3, 0.3)
    positions["C2"] = (4, 0.3)

    # User nodes (left of row 3-4)
    positions["U1"] = (0.3, 2.0)
    positions["U2"] = (0.3, 1.5)
    positions["U3"] = (0.3, 1.0)
    positions["U4"] = (0.3, 0.5)

    # Draw grid links
    edges = []
    for row in range(1, 5):
        for col in range(1, 5):
            src = f"R{row}{col}"
            if col < 4:
                dst = f"R{row}{col + 1}"
                edges.append((src, dst))
            if row < 4:
                dst = f"R{row + 1}{col}"
                edges.append((src, dst))

    # Access links
    access_edges = [
        ("E1", "R11"), ("E2", "R12"), ("E3", "R13"), ("E4", "R14"),
        ("E5", "R21"), ("E6", "R22"), ("E7", "R23"), ("E8", "R24"),
        ("C1", "R33"), ("C2", "R34"),
        ("U1", "R31"), ("U2", "R32"), ("U3", "R41"), ("U4", "R42"),
    ]

    # Draw all edges (light gray)
    for src, dst in edges + access_edges:
        if src in positions and dst in positions:
            x = [positions[src][0], positions[dst][0]]
            y = [positions[src][1], positions[dst][1]]
            ax.plot(x, y, "gray", linewidth=1, alpha=0.4)

    # Highlight congested link R32-R33 in red
    if "R32" in positions and "R33" in positions:
        x = [positions["R32"][0], positions["R33"][0]]
        y = [positions["R32"][1], positions["R33"][1]]
        ax.plot(x, y, "red", linewidth=3, alpha=0.6)

    # Highlight path
    if highlight_path:
        for i in range(len(highlight_path) - 1):
            src = highlight_path[i]
            dst = highlight_path[i + 1]
            if src in positions and dst in positions:
                x = [positions[src][0], positions[dst][0]]
                y = [positions[src][1], positions[dst][1]]
                ax.plot(x, y, path_color, linewidth=3, alpha=0.8)
                # Arrow
                dx = positions[dst][0] - positions[src][0]
                dy = positions[dst][1] - positions[src][1]
                mid_x = (positions[src][0] + positions[dst][0]) / 2
                mid_y = (positions[src][1] + positions[dst][1]) / 2
                ax.annotate(
                    "", xy=(mid_x + dx * 0.1, mid_y + dy * 0.1),
                    xytext=(mid_x - dx * 0.1, mid_y - dy * 0.1),
                    arrowprops=dict(
                        arrowstyle="->", color=path_color, lw=2,
                    ),
                )

    # Draw nodes
    for nid, (x, y) in positions.items():
        if nid.startswith("R"):
            ax.plot(x, y, "s", color="steelblue", markersize=14)
            ax.text(x, y, nid, ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        elif nid.startswith("E"):
            ax.plot(x, y, "o", color="orange", markersize=12)
            ax.text(x, y, nid, ha="center", va="center",
                    fontsize=7, fontweight="bold")
        elif nid.startswith("C"):
            ax.plot(x, y, "D", color="red", markersize=14)
            ax.text(x, y, nid, ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        elif nid.startswith("U"):
            ax.plot(x, y, "^", color="green", markersize=12)
            ax.text(x, y, nid, ha="center", va="center",
                    fontsize=6, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(color="steelblue", label="Core Router"),
        mpatches.Patch(color="orange", label="Edge Compute"),
        mpatches.Patch(color="red", label="Center Compute"),
        mpatches.Patch(color="green", label="User Node"),
        plt.Line2D([0], [0], color=path_color, linewidth=2,
                   label="Selected Path"),
        plt.Line2D([0], [0], color="red", linewidth=2, alpha=0.6,
                   label="Congested Link"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 6.0)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_all(exp1_dir: str = "results/exp1", exp2_dir: str = "results/exp2"):
    """Generate all figures from experiment results."""
    print("Generating figures...")

    exp1_results = os.path.join(exp1_dir, "exp1_results.json")
    exp2_results = os.path.join(exp2_dir, "exp2_results.json")

    if os.path.exists(exp1_results):
        plot_exp1_utilization(
            exp1_results,
            os.path.join(exp1_dir, "fig6_utilization_update.png"),
        )
    else:
        print(f"  Skipping Figure 6: {exp1_results} not found")

    if os.path.exists(exp2_results):
        plot_exp2_path_comparison(
            exp2_results,
            os.path.join(exp2_dir, "fig7_path_comparison.png"),
        )
    else:
        print(f"  Skipping Figure 7: {exp2_results} not found")

    print("Done.")


if __name__ == "__main__":
    plot_all()
