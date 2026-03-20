"""
Mininet Topology Script for SRv6 Computing-Aware Network Simulation.

Builds the 4x4 grid backbone with edge/center compute nodes and user nodes.
Requires: Mininet, BMv2 (behavioral-model), p4c compiler.

This script is intended to run in an environment with Mininet installed.
For standalone testing without Mininet, use utils/topology_builder.py instead.

Usage:
    sudo python topology/topo.py
"""

import os
import sys

# Mininet imports - these will only work in a Mininet-capable environment
try:
    from mininet.net import Mininet
    from mininet.topo import Topo
    from mininet.node import Host, Switch
    from mininet.link import TCLink
    from mininet.cli import CLI
    from mininet.log import setLogLevel, info

    MININET_AVAILABLE = True
except ImportError:
    MININET_AVAILABLE = False


class SRv6ComputeTopo(Topo):
    """SRv6 Computing-Aware Network Topology.

    4x4 grid of P4 switches with edge/center compute nodes and user nodes.
    """

    def build(self):
        """Build the topology."""
        info("*** Building SRv6 computing-aware topology\n")

        # --- Core routers: 4x4 grid of P4 switches ---
        routers = {}
        for row in range(1, 5):
            for col in range(1, 5):
                rid = f"R{row}{col}"
                routers[rid] = self.addSwitch(
                    rid,
                    cls=Switch,
                )
                info(f"  Added router {rid}\n")

        # Grid links (100 Mbps, 2ms delay)
        link_params = {"bw": 100, "delay": "2ms"}
        for row in range(1, 5):
            for col in range(1, 5):
                src = f"R{row}{col}"
                # Right neighbor
                if col < 4:
                    dst = f"R{row}{col + 1}"
                    self.addLink(routers[src], routers[dst], **link_params)
                # Down neighbor
                if row < 4:
                    dst = f"R{row + 1}{col}"
                    self.addLink(routers[src], routers[dst], **link_params)

        # --- Edge compute nodes: E1-E8 ---
        edge_connections = {
            "E1": "R11", "E2": "R12", "E3": "R13", "E4": "R14",
            "E5": "R21", "E6": "R22", "E7": "R23", "E8": "R24",
        }
        for eid, rid in edge_connections.items():
            host = self.addHost(eid)
            self.addLink(host, routers[rid], **link_params)
            info(f"  Added edge compute node {eid} -> {rid}\n")

        # --- Center compute nodes: C1, C2 ---
        center_connections = {"C1": "R33", "C2": "R34"}
        for cid, rid in center_connections.items():
            host = self.addHost(cid)
            self.addLink(host, routers[rid], **link_params)
            info(f"  Added center compute node {cid} -> {rid}\n")

        # --- User nodes: U1-U4 ---
        user_connections = {
            "U1": "R31", "U2": "R32", "U3": "R41", "U4": "R42",
        }
        for uid, rid in user_connections.items():
            host = self.addHost(uid)
            self.addLink(host, routers[rid], **link_params)
            info(f"  Added user node {uid} -> {rid}\n")


def run_topology():
    """Create and run the Mininet topology."""
    if not MININET_AVAILABLE:
        print("ERROR: Mininet is not installed in this environment.")
        print("Use 'utils/topology_builder.py' for standalone simulation.")
        sys.exit(1)

    setLogLevel("info")

    topo = SRv6ComputeTopo()
    net = Mininet(
        topo=topo,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True,
    )

    info("*** Starting network\n")
    net.start()

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    run_topology()
