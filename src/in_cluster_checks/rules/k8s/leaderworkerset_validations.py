"""
LeaderWorkerSet installation validations for OpenShift clusters.

Validates that the LeaderWorkerSet CRD is present in the cluster.
"""

from typing import ClassVar

from in_cluster_checks.core.rule import OrchestratorRule
from in_cluster_checks.core.rule_result import RuleResult
from in_cluster_checks.utils.enums import Objectives


class VerifyLeaderWorkerSetInstalled(OrchestratorRule):
    """Verify LeaderWorkerSet is installed by checking the required CRD exists.

    LeaderWorkerSet (LWS) manages multi-node inference deployments (e.g.
    disaggregated prefill/decode) for llm-d. This rule confirms the core
    CRD is registered, which indicates the LeaderWorkerSet operator has
    been installed on the cluster.
    """

    objective_hosts = [Objectives.ORCHESTRATOR]
    unique_name = "verify_leaderworkerset_installed"
    title = "Verify LeaderWorkerSet is installed"
    supported_profiles = {"llm-d-xks"}

    # LeaderWorkerSet CRD is namespaced
    REQUIRED_CRDS: ClassVar[list[tuple[str, bool]]] = [
        ("leaderworkersets.leaderworkerset.x-k8s.io", False),
    ]

    def run_rule(self) -> RuleResult:
        """Check that all required LeaderWorkerSet CRDs are present."""
        missing = [
            crd_name
            for crd_name, cluster_wide in self.REQUIRED_CRDS
            if not self.oc_api.crd_exists(crd_name, cluster_wide=cluster_wide)
        ]

        if missing:
            return RuleResult.failed("LeaderWorkerSet is not fully installed - missing CRDs: " + ", ".join(missing))

        return RuleResult.passed("All required LeaderWorkerSet CRDs are present")
