"""
Kuadrant installation validations for OpenShift clusters.

Validates that Kuadrant CRDs are present in the cluster.
"""

from typing import ClassVar

from in_cluster_checks.core.rule import OrchestratorRule
from in_cluster_checks.core.rule_result import RuleResult
from in_cluster_checks.utils.enums import Objectives


class VerifyKuadrantInstalled(OrchestratorRule):
    """Verify Kuadrant is installed by checking required CRDs exist.

    Kuadrant provides Gateway API policies for auth, rate limiting, and TLS.
    This rule confirms the core CRDs are registered, which indicates Kuadrant
    has been installed on the cluster. Missing CRDs produce a warning rather
    than a failure, since Kuadrant is optional for llm-d-xks deployments.
    """

    objective_hosts = [Objectives.ORCHESTRATOR]
    unique_name = "verify_kuadrant_installed"
    title = "Verify Kuadrant is installed"
    supported_profiles = {"llm-d-xks"}

    # Kuadrant CRDs are namespaced
    REQUIRED_CRDS: ClassVar[list[tuple[str, bool]]] = [
        ("kuadrants.kuadrant.io", False),
        ("authpolicies.kuadrant.io", False),
        ("ratelimitpolicies.kuadrant.io", False),
        ("tlspolicies.kuadrant.io", False),
        ("authconfigs.authorino.kuadrant.io", False),
        ("limitadors.limitador.kuadrant.io", False),
    ]

    def run_rule(self) -> RuleResult:
        """Check that all required Kuadrant CRDs are present."""
        missing = [
            crd_name
            for crd_name, cluster_wide in self.REQUIRED_CRDS
            if not self.oc_api.crd_exists(crd_name, cluster_wide=cluster_wide)
        ]

        if missing:
            return RuleResult.warning(
                "Kuadrant is not fully installed - missing CRDs: " + ", ".join(missing)
            )

        return RuleResult.passed("All required Kuadrant CRDs are present")
