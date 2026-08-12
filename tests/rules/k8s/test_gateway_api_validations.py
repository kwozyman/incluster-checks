"""Tests for Gateway API installation validations."""

from unittest.mock import Mock

import pytest

from in_cluster_checks.rules.k8s.gateway_api_validations import VerifyGatewayApiInstalled
from tests.pytest_tools.test_rule_base import RuleScenarioParams, RuleTestBase

_ALL_CRDS = [crd for crd, _ in VerifyGatewayApiInstalled.REQUIRED_CRDS]


def _crd_exists_side_effect(missing: set[str] | None = None):
    """Return a side_effect that reports the given CRDs as missing."""
    missing = missing or set()

    def _exists(resource_type: str, cluster_wide: bool = True, **_kwargs) -> bool:
        return resource_type not in missing

    return _exists


class TestVerifyGatewayApiInstalled(RuleTestBase):
    """Test VerifyGatewayApiInstalled rule."""

    tested_type = VerifyGatewayApiInstalled

    scenario_passed = [
        RuleScenarioParams(
            "all required Gateway API CRDs are present",
            tested_object_mock_dict={
                "oc_api.crd_exists": Mock(side_effect=_crd_exists_side_effect()),
            },
        ),
    ]

    scenario_failed = [
        RuleScenarioParams(
            "all required Gateway API CRDs are missing",
            tested_object_mock_dict={
                "oc_api.crd_exists": Mock(side_effect=_crd_exists_side_effect(set(_ALL_CRDS))),
            },
            failed_msg=("Gateway API is not fully installed - missing CRDs: " + ", ".join(_ALL_CRDS)),
        ),
        RuleScenarioParams(
            "some Gateway API CRDs are missing",
            tested_object_mock_dict={
                "oc_api.crd_exists": Mock(
                    side_effect=_crd_exists_side_effect({"httproutes.gateway.networking.k8s.io"})
                ),
            },
            failed_msg=(
                "Gateway API is not fully installed - missing CRDs: httproutes.gateway.networking.k8s.io"
            ),
        ),
    ]

    @pytest.mark.parametrize("scenario_params", scenario_passed)
    def test_scenario_passed(self, scenario_params, tested_object):
        """Test that rule passes when all Gateway API CRDs exist."""
        RuleTestBase.test_scenario_passed(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_failed)
    def test_scenario_failed(self, scenario_params, tested_object):
        """Test that rule fails when Gateway API CRDs are missing."""
        RuleTestBase.test_scenario_failed(self, scenario_params, tested_object)

    def test_checks_each_required_crd_with_correct_scope(self, tested_object):
        """Test that each required CRD is probed with the expected cluster_wide flag."""
        mock_crd_exists = Mock(return_value=True)
        tested_object.oc_api.crd_exists = mock_crd_exists

        result = tested_object.run_rule()

        assert result
        assert mock_crd_exists.call_count == len(VerifyGatewayApiInstalled.REQUIRED_CRDS)
        for crd_name, cluster_wide in VerifyGatewayApiInstalled.REQUIRED_CRDS:
            mock_crd_exists.assert_any_call(crd_name, cluster_wide=cluster_wide)
