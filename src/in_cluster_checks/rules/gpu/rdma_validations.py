"""
RDMA (InfiniBand/RoCE) hardware validations for AI/ML cluster nodes.

Ported from opendatahub-io/rhaii-cluster-validation's per-node RDMA checks
(pkg/checks/rdma/devices.go, pkg/checks/rdma/status.go), adapted to run via
`oc debug` instead of a privileged Kubernetes Job. GPU-NIC NUMA topology
pairing and the multi-node bandwidth/connectivity Jobs (iperf3, ib_write_bw,
ibv_rc_pingpong) from the upstream project are intentionally not ported.
"""

import re

from in_cluster_checks.core.rule import Rule
from in_cluster_checks.core.rule_result import PrerequisiteResult, RuleResult
from in_cluster_checks.utils.enums import Objectives
from in_cluster_checks.utils.safe_cmd_string import SafeCmdString

_CA_NAME_RE = re.compile(r"^CA '([^']+)'")
_PORT_RE = re.compile(r"Port (\d+):")
_STATE_RE = re.compile(r"State:\s+(\S+)")
_RATE_RE = re.compile(r"Rate:\s+(\S+)")
_LINK_LAYER_RE = re.compile(r"Link layer:\s+(\S+)")


def _extract_nic_fields(name: str, section: str) -> dict:
    """Extract State/Rate/Link layer fields for a single NIC (or NIC port) section."""
    state_match = _STATE_RE.search(section)
    rate_match = _RATE_RE.search(section)
    link_layer_match = _LINK_LAYER_RE.search(section)
    return {
        "name": name,
        "state": state_match.group(1) if state_match else "",
        "rate": rate_match.group(1) if rate_match else "",
        "link_layer": link_layer_match.group(1) if link_layer_match else "",
    }


def _parse_ibstat(output: str) -> list[dict]:
    """Parse `ibstat` text output into a list of {name, state, rate, link_layer} dicts.

    Multi-port CAs are expanded into one entry per port (e.g. "mlx5_0/port1").
    """
    nics = []
    for section in output.split("CA '")[1:]:
        name_match = _CA_NAME_RE.match("CA '" + section)
        if not name_match:
            continue
        ca_name = name_match.group(1)

        port_matches = list(_PORT_RE.finditer(section))
        if not port_matches:
            nics.append(_extract_nic_fields(ca_name, section))
            continue

        for i, port_match in enumerate(port_matches):
            start = port_match.end()
            end = port_matches[i + 1].start() if i + 1 < len(port_matches) else len(section)
            nics.append(_extract_nic_fields(f"{ca_name}/port{port_match.group(1)}", section[start:end]))

    return nics


class RdmaRule(Rule):
    """Base class for RDMA hardware rules.

    Only runs on nodes that expose the RDMA (InfiniBand/RoCE) subsystem.
    Nodes without RDMA NICs (e.g. pure-Ethernet/TCP inference clusters) are
    marked NOT_APPLICABLE rather than failed.
    """

    objective_hosts = [Objectives.ALL_NODES]
    supported_profiles = {"llm-d-xks"}

    def is_prerequisite_fulfilled(self) -> PrerequisiteResult:
        """Check that the RDMA subsystem is present on this node."""
        return_code, _, _ = self.run_cmd(SafeCmdString("test -d /sys/class/infiniband"))
        if return_code != 0:
            return PrerequisiteResult.not_met(
                "No RDMA subsystem present on this node (/sys/class/infiniband not found)"
            )
        return PrerequisiteResult.met()


class VerifyRdmaDevicesPresent(RdmaRule):
    """Verify RDMA devices are present and accessible on the node."""

    unique_name = "verify_rdma_devices_present"
    title = "Verify RDMA devices are detected and accessible"

    def run_rule(self) -> RuleResult:
        """Check /dev/infiniband is accessible and at least one RDMA device is registered."""
        return_code, _, _ = self.run_cmd(SafeCmdString("test -e /dev/infiniband"))
        if return_code != 0:
            return RuleResult.failed("/dev/infiniband not found - enable RDMA networking on this node")

        return_code, out, err = self.run_cmd(SafeCmdString("ls /sys/class/infiniband"))
        if return_code != 0:
            return RuleResult.failed(f"Failed to enumerate RDMA devices: {(err or out).strip()}")

        devices = out.split()
        if not devices:
            return RuleResult.failed(
                "No RDMA devices found under /sys/class/infiniband - "
                "check RDMA device plugin and network operator installation"
            )

        return RuleResult.passed(f"{len(devices)} RDMA device(s) detected: {', '.join(devices)}")


class VerifyRdmaNicStatus(RdmaRule):
    """Verify RDMA NIC link state is active."""

    unique_name = "verify_rdma_nic_status"
    title = "Verify RDMA NIC link status is active"

    def run_rule(self) -> RuleResult:
        """Run ibstat and check that RDMA NICs report an Active link state."""
        return_code, out, err = self.run_cmd(SafeCmdString("ibstat"))
        if return_code != 0:
            return RuleResult.failed(
                f"ibstat failed: {(err or out).strip()} - check RDMA driver and device plugin installation"
            )

        nics = _parse_ibstat(out)
        if not nics:
            return RuleResult.failed("No RDMA NICs found via ibstat")

        active = [f"{nic['name']} ({nic['rate']} Gbps)" for nic in nics if nic["state"] == "Active"]
        down = [nic["name"] for nic in nics if nic["state"] != "Active"]

        if down and not active:
            return RuleResult.failed(
                f"All {len(down)} RDMA NIC(s) down: {', '.join(down)} - check NIC, cable, and switch configuration"
            )
        if down:
            return RuleResult.warning(
                f"{len(down)}/{len(nics)} RDMA NIC(s) down: {', '.join(down)}; active: {', '.join(active)}"
            )

        return RuleResult.passed(f"{len(active)} active RDMA NIC(s): {', '.join(active)}")
