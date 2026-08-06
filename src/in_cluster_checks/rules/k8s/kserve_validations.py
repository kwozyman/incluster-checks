"""
KServe installation validations for OpenShift clusters.

Validates that KServe and inference networking CRDs are present in the cluster.
"""

from typing import ClassVar

from in_cluster_checks.core.rule import OrchestratorRule
from in_cluster_checks.core.rule_result import RuleResult
from in_cluster_checks.utils.enums import Objectives


class VerifyKServeInstalled(OrchestratorRule):
    """Verify KServe is installed by checking required CRDs exist.

    KServe provides LLM inference serving on Kubernetes. This rule confirms
    the KServe and inference networking CRDs are registered, which indicates
    KServe has been installed on the cluster.
    """

    objective_hosts = [Objectives.ORCHESTRATOR]
    unique_name = "verify_kserve_installed"
    title = "Verify KServe is installed"
    supported_profiles = {"llm-d-xks"}

    # KServe and inference networking CRDs are namespaced
    REQUIRED_CRDS: ClassVar[list[tuple[str, bool]]] = [
        ("llminferenceservices.serving.kserve.io", False),
        ("llminferenceserviceconfigs.serving.kserve.io", False),
        ("inferencepools.inference.networking.k8s.io", False),
        ("inferencemodels.inference.networking.x-k8s.io", False),
        ("inferenceobjectives.inference.networking.x-k8s.io", False),
        ("inferencepoolimports.inference.networking.x-k8s.io", False),
        ("inferencepools.inference.networking.x-k8s.io", False),
    ]

    def run_rule(self) -> RuleResult:
        """Check that all required KServe CRDs are present."""
        missing = [
            crd_name
            for crd_name, cluster_wide in self.REQUIRED_CRDS
            if not self.oc_api.crd_exists(crd_name, cluster_wide=cluster_wide)
        ]

        if missing:
            return RuleResult.failed(
                "KServe is not fully installed - missing CRDs: " + ", ".join(missing)
            )

        return RuleResult.passed("All required KServe CRDs are present")
