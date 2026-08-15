"""Tests for GPU/RDMA validation domain."""

from in_cluster_checks.domains.gpu_domain import GpuValidationDomain
from in_cluster_checks.rules.gpu.gpu_validations import (
    VerifyGpuDriverInstalled,
    VerifyGpuEccErrorsAbsent,
    VerifyGpuNodeLabelPresent,
)
from in_cluster_checks.rules.gpu.rdma_validations import (
    VerifyRdmaDevicesPresent,
    VerifyRdmaNicStatus,
)


def test_gpu_domain_name():
    """Test domain name."""
    domain = GpuValidationDomain()
    assert domain.domain_name() == "gpu"


def test_gpu_domain_rules():
    """Test domain returns correct rules."""
    domain = GpuValidationDomain()
    rules = domain.get_rule_classes()

    assert len(rules) == 5
    assert VerifyGpuNodeLabelPresent in rules
    assert VerifyGpuDriverInstalled in rules
    assert VerifyGpuEccErrorsAbsent in rules
    assert VerifyRdmaDevicesPresent in rules
    assert VerifyRdmaNicStatus in rules
