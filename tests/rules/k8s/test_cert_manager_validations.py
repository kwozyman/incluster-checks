"""Tests for cert-manager installation validations."""

from unittest.mock import Mock

import pytest

from in_cluster_checks.rules.k8s.cert_manager_validations import VerifyCertManagerInstalled
from tests.pytest_tools.test_rule_base import RuleScenarioParams, RuleTestBase

_ALL_CRDS = [crd for crd, _ in VerifyCertManagerInstalled.REQUIRED_CRDS]


def _crd_exists_side_effect(missing: set[str] | None = None):
    """Return a side_effect that reports the given CRDs as missing."""
    missing = missing or set()

    def _exists(resource_type: str, cluster_wide: bool = True, **_kwargs) -> bool:
        return resource_type not in missing

    return _exists


class TestVerifyCertManagerInstalled(RuleTestBase):
    """Test VerifyCertManagerInstalled rule."""

    tested_type = VerifyCertManagerInstalled

    scenario_passed = [
        RuleScenarioParams(
            "all required cert-manager CRDs are present",
            tested_object_mock_dict={
                "oc_api.crd_exists": Mock(side_effect=_crd_exists_side_effect()),
            },
        ),
    ]

    scenario_failed = [
        RuleScenarioParams(
            "all required cert-manager CRDs are missing",
            tested_object_mock_dict={
                "oc_api.crd_exists": Mock(side_effect=_crd_exists_side_effect(set(_ALL_CRDS))),
            },
            failed_msg=(
                "cert-manager is not fully installed - missing CRDs: "
                + ", ".join(_ALL_CRDS)
            ),
        ),
        RuleScenarioParams(
            "some cert-manager CRDs are missing",
            tested_object_mock_dict={
                "oc_api.crd_exists": Mock(
                    side_effect=_crd_exists_side_effect(
                        {"clusterissuers.cert-manager.io", "issuers.cert-manager.io"}
                    )
                ),
            },
            failed_msg=(
                "cert-manager is not fully installed - missing CRDs: "
                "clusterissuers.cert-manager.io, issuers.cert-manager.io"
            ),
        ),
    ]

    @pytest.mark.parametrize("scenario_params", scenario_passed)
    def test_scenario_passed(self, scenario_params, tested_object):
        """Test that rule passes when all cert-manager CRDs exist."""
        RuleTestBase.test_scenario_passed(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_failed)
    def test_scenario_failed(self, scenario_params, tested_object):
        """Test that rule fails when cert-manager CRDs are missing."""
        RuleTestBase.test_scenario_failed(self, scenario_params, tested_object)

    def test_checks_each_required_crd_with_correct_scope(self, tested_object):
        """Test that each required CRD is probed with the expected cluster_wide flag."""
        mock_crd_exists = Mock(return_value=True)
        tested_object.oc_api.crd_exists = mock_crd_exists

        result = tested_object.run_rule()

        assert result
        assert mock_crd_exists.call_count == len(VerifyCertManagerInstalled.REQUIRED_CRDS)
        for crd_name, cluster_wide in VerifyCertManagerInstalled.REQUIRED_CRDS:
            mock_crd_exists.assert_any_call(crd_name, cluster_wide=cluster_wide)
