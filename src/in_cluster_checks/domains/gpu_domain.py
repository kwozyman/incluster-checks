"""
GPU/RDMA rule domain for AI/ML cluster nodes.

Orchestrates GPU hardware and RDMA networking healthcheck validators,
ported from opendatahub-io/rhaii-cluster-validation.
"""

from typing import List

from in_cluster_checks.core.domain import RuleDomain
from in_cluster_checks.rules.gpu.gpu_validations import VerifyGpuDriverInstalled, VerifyGpuEccErrorsAbsent
from in_cluster_checks.rules.gpu.rdma_validations import VerifyRdmaDevicesPresent, VerifyRdmaNicStatus


class GpuValidationDomain(RuleDomain):
    """
    GPU/RDMA rule domain.

    Validates GPU hardware (driver, ECC) and RDMA networking health on
    cluster nodes used for AI/ML inference workloads.
    """

    def domain_name(self) -> str:
        """Get domain name."""
        return "gpu"

    def get_rule_classes(self) -> List[type]:
        """
        Get list of GPU/RDMA validators to run.

        Returns:
            List of Rule classes
        """
        return [
            VerifyGpuDriverInstalled,
            VerifyGpuEccErrorsAbsent,
            VerifyRdmaDevicesPresent,
            VerifyRdmaNicStatus,
        ]
