"""
cert-manager installation validations for OpenShift clusters.

Validates that cert-manager CRDs are present in the cluster.
"""

from typing import ClassVar

from in_cluster_checks.core.rule import OrchestratorRule
from in_cluster_checks.core.rule_result import RuleResult
from in_cluster_checks.utils.enums import Objectives


class VerifyCertManagerInstalled(OrchestratorRule):
    """Verify cert-manager is installed by checking required CRDs exist.

    cert-manager provides certificate management for Kubernetes. This rule
    confirms the core CRDs are registered, which indicates cert-manager has
    been installed on the cluster.
    """

    objective_hosts = [Objectives.ORCHESTRATOR]
    unique_name = "verify_cert_manager_installed"
    title = "Verify cert-manager is installed"
    supported_profiles = {"llm-d-xks"}

    # (resource type, cluster_wide) — ClusterIssuer is cluster-scoped; others are namespaced
    REQUIRED_CRDS: ClassVar[list[tuple[str, bool]]] = [
        ("certificaterequests.cert-manager.io", False),
        ("certificates.cert-manager.io", False),
        ("clusterissuers.cert-manager.io", True),
        ("issuers.cert-manager.io", False),
    ]

    def run_rule(self) -> RuleResult:
        """Check that all required cert-manager CRDs are present."""
        missing = [
            crd_name
            for crd_name, cluster_wide in self.REQUIRED_CRDS
            if not self.oc_api.crd_exists(crd_name, cluster_wide=cluster_wide)
        ]

        if missing:
            return RuleResult.failed(
                "cert-manager is not fully installed - missing CRDs: " + ", ".join(missing)
            )

        return RuleResult.passed("All required cert-manager CRDs are present")
