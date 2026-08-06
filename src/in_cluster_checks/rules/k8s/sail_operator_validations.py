"""
Sail Operator installation validations for OpenShift clusters.

Validates that Sail Operator CRDs are present in the cluster.
"""

from typing import ClassVar

from in_cluster_checks.core.rule import OrchestratorRule
from in_cluster_checks.core.rule_result import RuleResult
from in_cluster_checks.utils.enums import Objectives


class VerifySailOperatorInstalled(OrchestratorRule):
    """Verify Sail Operator is installed by checking required CRDs exist.

    Sail Operator manages the Istio control plane lifecycle. This rule
    confirms the core CRDs are registered, which indicates Sail Operator
    has been installed on the cluster.
    """

    objective_hosts = [Objectives.ORCHESTRATOR]
    unique_name = "verify_sail_operator_installed"
    title = "Verify Sail Operator is installed"
    supported_profiles = {"llm-d-xks"}

    # All Sail Operator CRDs are cluster-scoped
    REQUIRED_CRDS: ClassVar[list[tuple[str, bool]]] = [
        ("istiocnis.sailoperator.io", True),
        ("istiorevisions.sailoperator.io", True),
        ("istiorevisiontags.sailoperator.io", True),
        ("istios.sailoperator.io", True),
        ("ztunnels.sailoperator.io", True),
    ]

    def run_rule(self) -> RuleResult:
        """Check that all required Sail Operator CRDs are present."""
        missing = [
            crd_name
            for crd_name, cluster_wide in self.REQUIRED_CRDS
            if not self.oc_api.crd_exists(crd_name, cluster_wide=cluster_wide)
        ]

        if missing:
            return RuleResult.failed(
                "Sail Operator is not fully installed - missing CRDs: " + ", ".join(missing)
            )

        return RuleResult.passed("All required Sail Operator CRDs are present")
