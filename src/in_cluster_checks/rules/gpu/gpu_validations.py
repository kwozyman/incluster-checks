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
from in_cluster_checks.utils.safe_cmd_string import SafeCmdString

# Values reported by nvidia-smi for ECC error counters that don't indicate a real error
_NON_ERROR_ECC_VALUES = {"0", "N/A", ""}


def _parse_csv_rows(output: str) -> list[list[str]]:
    """Parse nvidia-smi CSV output (one row per GPU) into a list of trimmed fields."""
    reader = csv.reader(io.StringIO(output.strip()))
    return [[field.strip() for field in row] for row in reader if row]


class NvidiaGpuRule(Rule):
    """Base class for NVIDIA GPU hardware rules.

    Only runs on nodes where `nvidia-smi` is available. AMD GPU nodes
    (rocm-smi) are not covered by this rule - see rhaii-cluster-validation
    for a reference implementation if AMD support is needed.
    """

    objective_hosts = [Objectives.ALL_NODES]
    supported_profiles = {"llm-d-xks"}

    def is_prerequisite_fulfilled(self) -> PrerequisiteResult:
        """Check that nvidia-smi is available on this node."""
        return_code, _, _ = self.run_cmd(SafeCmdString("which nvidia-smi"))
        if return_code != 0:
            return PrerequisiteResult.not_met("nvidia-smi is not available on this node (not an NVIDIA GPU node)")
        return PrerequisiteResult.met()


class VerifyGpuDriverInstalled(NvidiaGpuRule):
    """Verify the NVIDIA GPU driver is installed and reporting correctly."""

    unique_name = "verify_gpu_driver_installed"
    title = "Verify NVIDIA GPU driver is installed and functional"

    DRIVER_QUERY_CMD = SafeCmdString(
        "nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader,nounits"
    )

    def run_rule(self) -> RuleResult:
        """Run nvidia-smi and validate the driver reports GPU information."""
        return_code, out, err = self.run_cmd(self.DRIVER_QUERY_CMD)
        if return_code != 0:
            return RuleResult.failed(f"nvidia-smi failed: {(err or out).strip()}")

        try:
            rows = _parse_csv_rows(out)
        except csv.Error as e:
            return RuleResult.failed(f"Failed to parse nvidia-smi driver output: {e}")

        if not rows or len(rows[0]) < 3:
            return RuleResult.failed(f"nvidia-smi returned no GPU driver information: {out.strip()!r}")

        driver_version, gpu_name, memory_total = rows[0][:3]
        gpu_count = len(rows)

        return RuleResult.passed(
            f"NVIDIA driver: {driver_version}, GPU: {gpu_name} ({memory_total} MiB), {gpu_count} GPU(s)"
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
        return_code, out, err = self.run_cmd(self.ECC_QUERY_CMD)
        if return_code != 0:
            # Some GPUs/drivers don't support ECC error queries - treat as a warning, not a failure.
            return RuleResult.warning(f"nvidia-smi ECC query failed: {(err or out).strip()}")

        try:
            rows = _parse_csv_rows(out)
        except csv.Error as e:
            return RuleResult.failed(f"Failed to parse nvidia-smi ECC output: {e}")

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

        return RuleResult.passed(f"No uncorrectable ECC errors on {len(rows)} GPU(s)")
