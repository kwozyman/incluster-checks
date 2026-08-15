"""
NVIDIA GPU hardware validations for AI/ML cluster nodes.

Ported from opendatahub-io/rhaii-cluster-validation's per-node GPU checks
(pkg/checks/gpu/driver.go, pkg/checks/gpu/ecc.go), adapted to run via
`oc debug` instead of a privileged Kubernetes Job.
"""

import csv
import io

from in_cluster_checks.core.rule import Rule
from in_cluster_checks.core.rule_result import PrerequisiteResult, RuleResult
from in_cluster_checks.utils.enums import Objectives
from in_cluster_checks.utils.oc_api_utils import OcApiUtils
from in_cluster_checks.utils.safe_cmd_string import SafeCmdString

# Values reported by nvidia-smi for ECC error counters that don't indicate a real error
_NON_ERROR_ECC_VALUES = {"0", "N/A", ""}

# Node label indicating NVIDIA GPU hardware presence (set by the NVIDIA GPU
# Operator / Node Feature Discovery).
GPU_PRESENCE_LABEL = "nvidia.com/gpu.present"
# Node allocatable resource indicating schedulable NVIDIA GPUs (set by the device plugin)
GPU_RESOURCE_NAME = "nvidia.com/gpu"
# Labels of NVIDIA GPU Operator pods that have nvidia-smi available via the NVIDIA
# Container Toolkit runtime hook, checked in this order. The device plugin pod is
# checked first since it's present even when the driver is pre-installed on the host
# (e.g. GPU-optimized AMIs) and no driver daemonset pod is deployed at all.
GPU_OPERATOR_POD_LABELS = (
    {"app": "nvidia-device-plugin-daemonset"},
    {"app": "nvidia-driver-daemonset"},
)


def _parse_csv_rows(output: str) -> list[list[str]]:
    """Parse nvidia-smi CSV output (one row per GPU) into a list of trimmed fields."""
    reader = csv.reader(io.StringIO(output.strip()))
    return [[field.strip() for field in row] for row in reader if row]


def _gpu_quantity(value) -> int:
    """Parse a node allocatable/capacity GPU resource quantity (always a plain integer count)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_node_gpu_signals(oc_api: OcApiUtils, node_name: str) -> tuple[bool, bool, int]:
    """Look up a node via the cluster API and read its GPU label/resource signals.

    Args:
        oc_api: Cluster API accessor
        node_name: Name of the node to look up

    Returns:
        Tuple of (node_found, has_gpu_label, gpu_resource_count)
    """
    node = oc_api.select_resources(f"node/{node_name}", single=True)
    if not node:
        return False, False, 0

    labels = node.model.metadata.labels or {}
    allocatable = node.model.status.allocatable or {}

    has_gpu_label = labels.get(GPU_PRESENCE_LABEL) == "true"
    gpu_resource_count = _gpu_quantity(allocatable.get(GPU_RESOURCE_NAME))
    return True, has_gpu_label, gpu_resource_count


class VerifyGpuNodeLabelPresent(Rule):
    """Verify the node is labeled as a GPU node.

    This is a standalone check of the Kubernetes-level GPU label/resource
    state (set by the NVIDIA GPU Operator / Node Feature Discovery),
    reported independently of whether nvidia-smi is reachable from within
    the debug pod - see VerifyGpuDriverInstalled/VerifyGpuEccErrorsAbsent
    below for the nvidia-smi based checks.
    """

    objective_hosts = [Objectives.ALL_NODES]
    supported_profiles = {"llm-d-xks"}
    unique_name = "verify_gpu_node_label_present"
    title = "Verify node is labeled as a GPU node"

    def __init__(self, host_executor, node_executors=None):
        """Initialize the rule and its cluster API accessor (for node label/resource lookups)."""
        super().__init__(host_executor, node_executors=node_executors)
        self.oc_api = OcApiUtils(self)

    def run_rule(self) -> RuleResult:
        """Check the node's GPU label/resource state and report accordingly."""
        node_name = self.get_host_name()
        found, has_gpu_label, gpu_resource_count = _get_node_gpu_signals(self.oc_api, node_name)

        if not found:
            return RuleResult.failed(f"Node {node_name} could not be found via the cluster API")

        if has_gpu_label:
            return RuleResult.passed(f"Node is labeled as a GPU node ({GPU_PRESENCE_LABEL}=true)")

        if gpu_resource_count > 0:
            return RuleResult.warning(
                f"Node has {gpu_resource_count} schedulable {GPU_RESOURCE_NAME} resource(s) "
                f"but is missing the {GPU_PRESENCE_LABEL} label"
            )

        return RuleResult.not_applicable(f"Node {node_name} has no GPU labels/resources (not a GPU node)")


class NvidiaGpuRule(Rule):
    """Base class for NVIDIA GPU hardware rules.

    Applicability is determined by NVIDIA GPU labels/resources on the node
    object (set by the NVIDIA GPU Operator / Node Feature Discovery), not
    by probing for the nvidia-smi binary - some driver-container installs
    don't expose nvidia-smi via `chroot /host`. See VerifyGpuNodeLabelPresent
    above for a standalone report of the label/resource state itself.

    nvidia-smi is run directly on the node when reachable via `chroot /host`.
    Otherwise, subclasses fall back to running it inside an NVIDIA GPU
    Operator pod (device plugin or driver daemonset) scheduled on this node,
    via `oc exec` (no shell involved, since these are typically minimal
    containers that may not have bash) - the NVIDIA Container Toolkit makes
    nvidia-smi available in those pods regardless of how the driver itself
    was installed. If nvidia-smi can't be reached either way, subclasses
    report a WARNING rather than a hard failure or NOT_APPLICABLE.

    AMD GPU nodes (rocm-smi) are not covered by this rule - see
    rhaii-cluster-validation for a reference implementation if AMD support
    is needed.
    """

    objective_hosts = [Objectives.ALL_NODES]
    supported_profiles = {"llm-d-xks"}

    NVIDIA_SMI_NOT_AVAILABLE_MSG = (
        "nvidia-smi is not reachable on this node - neither directly (via the debug pod) nor "
        "inside an NVIDIA GPU Operator pod (device plugin or driver daemonset) scheduled on "
        "this node - even though the node was identified as a GPU node from its labels/resources"
    )

    def __init__(self, host_executor, node_executors=None):
        """Initialize the rule and its cluster API accessor (for node label/resource lookups)."""
        super().__init__(host_executor, node_executors=node_executors)
        self.oc_api = OcApiUtils(self)

    def is_prerequisite_fulfilled(self) -> PrerequisiteResult:
        """Check that this node is a GPU node, based on its labels/resources."""
        node_name = self.get_host_name()
        found, has_gpu_label, gpu_resource_count = _get_node_gpu_signals(self.oc_api, node_name)

        if not found:
            return PrerequisiteResult.not_met(f"Node {node_name} could not be found via the cluster API")

        if not (has_gpu_label or gpu_resource_count > 0):
            return PrerequisiteResult.not_met(f"Node {node_name} has no GPU labels/resources (not a GPU node)")

        return PrerequisiteResult.met()

    def _is_nvidia_smi_available(self) -> bool:
        """Check whether the nvidia-smi binary is reachable directly on this node."""
        return_code, _, _ = self.run_cmd(SafeCmdString("which nvidia-smi"))
        return return_code == 0

    def _find_gpu_operator_pod(self) -> tuple[str, str] | None:
        """Find an NVIDIA GPU Operator pod running on this node that has nvidia-smi
        available (device plugin or driver daemonset - see GPU_OPERATOR_POD_LABELS).

        Returns:
            (namespace, pod_name) tuple of a Running GPU Operator pod on this node, or
            None if no such pod could be found (e.g. the GPU Operator isn't installed).
        """
        node_name = self.get_host_name()
        for labels in GPU_OPERATOR_POD_LABELS:
            try:
                pods = self.oc_api.get_pods(labels=labels)
            except Exception as e:
                self.logger.debug(f"Failed to search for GPU Operator pods with labels {labels}: {e}")
                continue

            for pod in pods:
                if pod.model.spec.nodeName == node_name and pod.model.status.phase == "Running":
                    return pod.namespace(), pod.name()

        return None

    def _run_nvidia_smi(self, cmd: SafeCmdString) -> tuple[int, str, str, str] | None:
        """Run an nvidia-smi command, preferring the node itself and falling back to an
        NVIDIA GPU Operator pod scheduled on this node.

        Args:
            cmd: nvidia-smi command to run

        Returns:
            (return_code, stdout, stderr, source) tuple, where source describes where the
            command ran, or None if nvidia-smi could not be reached anywhere.
        """
        if self._is_nvidia_smi_available():
            return_code, out, err = self.run_cmd(cmd)
            return return_code, out, err, "on the node"

        gpu_operator_pod = self._find_gpu_operator_pod()
        if gpu_operator_pod is None:
            return None

        namespace, pod_name = gpu_operator_pod
        return_code, out, err = self.oc_api.run_exec_cmd(namespace, pod_name, cmd)
        return return_code, out, err, f"in GPU Operator pod {namespace}/{pod_name}"


class VerifyGpuDriverInstalled(NvidiaGpuRule):
    """Verify the NVIDIA GPU driver is installed and reporting correctly."""

    unique_name = "verify_gpu_driver_installed"
    title = "Verify NVIDIA GPU driver is installed and functional"

    DRIVER_QUERY_CMD = SafeCmdString(
        "nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader,nounits"
    )

    def run_rule(self) -> RuleResult:
        """Run nvidia-smi and validate the driver reports GPU information."""
        result = self._run_nvidia_smi(self.DRIVER_QUERY_CMD)
        if result is None:
            return RuleResult.warning(self.NVIDIA_SMI_NOT_AVAILABLE_MSG)

        return_code, out, err, source = result
        if return_code != 0:
            return RuleResult.failed(f"nvidia-smi failed ({source}): {(err or out).strip()}")

        try:
            rows = _parse_csv_rows(out)
        except csv.Error as e:
            return RuleResult.failed(f"Failed to parse nvidia-smi driver output ({source}): {e}")

        if not rows or len(rows[0]) < 3:
            return RuleResult.failed(f"nvidia-smi returned no GPU driver information ({source}): {out.strip()!r}")

        driver_version, gpu_name, memory_total = rows[0][:3]
        gpu_count = len(rows)

        return RuleResult.passed(
            f"NVIDIA driver: {driver_version}, GPU: {gpu_name} ({memory_total} MiB), "
            f"{gpu_count} GPU(s) (checked {source})"
        )


class VerifyGpuEccErrorsAbsent(NvidiaGpuRule):
    """Verify no uncorrectable ECC memory errors are reported on any GPU."""

    unique_name = "verify_gpu_ecc_errors_absent"
    title = "Verify GPU ECC memory has no uncorrectable errors"

    ECC_QUERY_CMD = SafeCmdString(
        "nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits"
    )

    def run_rule(self) -> RuleResult:
        """Run nvidia-smi ECC query and check for uncorrectable errors."""
        result = self._run_nvidia_smi(self.ECC_QUERY_CMD)
        if result is None:
            return RuleResult.warning(self.NVIDIA_SMI_NOT_AVAILABLE_MSG)

        return_code, out, err, source = result
        if return_code != 0:
            # Some GPUs/drivers don't support ECC error queries - treat as a warning, not a failure.
            return RuleResult.warning(f"nvidia-smi ECC query failed ({source}): {(err or out).strip()}")

        try:
            rows = _parse_csv_rows(out)
        except csv.Error as e:
            return RuleResult.failed(f"Failed to parse nvidia-smi ECC output ({source}): {e}")

        errors_found = []
        for row in rows:
            if len(row) < 2:
                continue
            gpu_index, ecc_errors = row[0], row[1]
            if ecc_errors not in _NON_ERROR_ECC_VALUES:
                errors_found.append(f"GPU {gpu_index}: {ecc_errors} uncorrectable errors")

        if errors_found:
            return RuleResult.failed(
                "Uncorrectable ECC errors found: " + "; ".join(errors_found) + ". Replace GPU or contact cloud provider"
            )

        return RuleResult.passed(f"No uncorrectable ECC errors on {len(rows)} GPU(s) (checked {source})")
