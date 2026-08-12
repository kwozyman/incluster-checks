"""
Gateway API installation validations for OpenShift clusters.

Validates that Gateway API CRDs are present in the cluster.
"""

from typing import ClassVar

from in_cluster_checks.core.rule import OrchestratorRule
from in_cluster_checks.core.rule_result import RuleResult
from in_cluster_checks.utils.enums import Objectives


class VerifyGatewayApiInstalled(OrchestratorRule):
    """Verify Gateway API is installed by checking required CRDs exist.

    Gateway API provides the Gateway and HTTPRoute resources used to expose
    llm-d inference services. This rule confirms the core CRDs are
    registered, which indicates Gateway API has been installed on the
    cluster.
    """

    objective_hosts = [Objectives.ORCHESTRATOR]
    unique_name = "verify_gateway_api_installed"
    title = "Verify Gateway API is installed"
    supported_profiles = {"llm-d-xks"}

    # Gateway API CRDs are namespaced
    REQUIRED_CRDS: ClassVar[list[tuple[str, bool]]] = [
        ("gateways.gateway.networking.k8s.io", False),
        ("httproutes.gateway.networking.k8s.io", False),
    ]

    def run_rule(self) -> RuleResult:
        """Check that all required Gateway API CRDs are present."""
        missing = [
            crd_name
            for crd_name, cluster_wide in self.REQUIRED_CRDS
            if not self.oc_api.crd_exists(crd_name, cluster_wide=cluster_wide)
        ]

        if missing:
            return RuleResult.failed("Gateway API is not fully installed - missing CRDs: " + ", ".join(missing))

        return RuleResult.passed("All required Gateway API CRDs are present")
