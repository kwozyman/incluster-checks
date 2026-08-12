"""Tests for RDMA hardware validations."""

import pytest

from in_cluster_checks.rules.gpu.rdma_validations import (
    VerifyRdmaDevicesPresent,
    VerifyRdmaNicStatus,
)
from tests.pytest_tools.test_operator_base import CmdOutput
from tests.pytest_tools.test_rule_base import RuleScenarioParams, RuleTestBase

RDMA_SYSFS_PRESENT_CMD = "test -d /sys/class/infiniband"
RDMA_DEV_NODE_CMD = "test -e /dev/infiniband"
LIST_RDMA_DEVICES_CMD = "ls /sys/class/infiniband"
IBSTAT_CMD = "ibstat"

IBSTAT_ALL_ACTIVE = """CA 'mlx5_0'
\tCA type: MT4129
\tNumber of ports: 1
\tFirmware version: 28.36.1010
\tPort 1:
\t\tState: Active
\t\tPhysical state: LinkUp
\t\tRate: 200
\t\tLink layer: InfiniBand
"""

IBSTAT_ONE_DOWN = """CA 'mlx5_0'
\tCA type: MT4129
\tNumber of ports: 1
\tPort 1:
\t\tState: Active
\t\tPhysical state: LinkUp
\t\tRate: 200
\t\tLink layer: InfiniBand
CA 'mlx5_1'
\tCA type: MT4129
\tNumber of ports: 1
\tPort 1:
\t\tState: Down
\t\tPhysical state: Disabled
\t\tRate: 10
\t\tLink layer: InfiniBand
"""

IBSTAT_ALL_DOWN = """CA 'mlx5_0'
\tCA type: MT4129
\tNumber of ports: 1
\tPort 1:
\t\tState: Down
\t\tPhysical state: Disabled
\t\tRate: 10
\t\tLink layer: InfiniBand
"""


class TestVerifyRdmaDevicesPresent(RuleTestBase):
    """Test VerifyRdmaDevicesPresent rule."""

    tested_type = VerifyRdmaDevicesPresent

    scenario_passed = [
        RuleScenarioParams(
            "RDMA devices present and accessible",
            {
                RDMA_SYSFS_PRESENT_CMD: CmdOutput(""),
                RDMA_DEV_NODE_CMD: CmdOutput(""),
                LIST_RDMA_DEVICES_CMD: CmdOutput("mlx5_0  mlx5_1"),
            },
        ),
    ]

    scenario_failed = [
        RuleScenarioParams(
            "/dev/infiniband missing",
            {
                RDMA_SYSFS_PRESENT_CMD: CmdOutput(""),
                RDMA_DEV_NODE_CMD: CmdOutput("", return_code=1),
            },
            failed_msg="/dev/infiniband not found - enable RDMA networking on this node",
        ),
        RuleScenarioParams(
            "no RDMA devices registered",
            {
                RDMA_SYSFS_PRESENT_CMD: CmdOutput(""),
                RDMA_DEV_NODE_CMD: CmdOutput(""),
                LIST_RDMA_DEVICES_CMD: CmdOutput(""),
            },
            failed_msg=(
                "No RDMA devices found under /sys/class/infiniband - "
                "check RDMA device plugin and network operator installation"
            ),
        ),
    ]

    scenario_not_applicable = [
        RuleScenarioParams(
            "no RDMA subsystem on this node",
            {
                RDMA_SYSFS_PRESENT_CMD: CmdOutput("", return_code=1),
            },
        ),
    ]

    @pytest.mark.parametrize("scenario_params", scenario_passed)
    def test_scenario_passed(self, scenario_params, tested_object):
        RuleTestBase.test_scenario_passed(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_failed)
    def test_scenario_failed(self, scenario_params, tested_object):
        RuleTestBase.test_scenario_failed(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_passed)
    def test_prerequisite_fulfilled(self, scenario_params, tested_object):
        RuleTestBase.test_prerequisite_fulfilled(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_not_applicable)
    def test_prerequisite_not_fulfilled(self, scenario_params, tested_object):
        RuleTestBase.test_prerequisite_not_fulfilled(self, scenario_params, tested_object)


class TestVerifyRdmaNicStatus(RuleTestBase):
    """Test VerifyRdmaNicStatus rule."""

    tested_type = VerifyRdmaNicStatus

    scenario_passed = [
        RuleScenarioParams(
            "all RDMA NICs active",
            {
                RDMA_SYSFS_PRESENT_CMD: CmdOutput(""),
                IBSTAT_CMD: CmdOutput(IBSTAT_ALL_ACTIVE),
            },
        ),
    ]

    scenario_failed = [
        RuleScenarioParams(
            "ibstat command fails",
            {
                RDMA_SYSFS_PRESENT_CMD: CmdOutput(""),
                IBSTAT_CMD: CmdOutput("", return_code=1, err="command not found"),
            },
            failed_msg=(
                "ibstat failed: command not found - check RDMA driver and device plugin installation"
            ),
        ),
        RuleScenarioParams(
            "all RDMA NICs down",
            {
                RDMA_SYSFS_PRESENT_CMD: CmdOutput(""),
                IBSTAT_CMD: CmdOutput(IBSTAT_ALL_DOWN),
            },
            failed_msg=(
                "All 1 RDMA NIC(s) down: mlx5_0/port1 - check NIC, cable, and switch configuration"
            ),
        ),
    ]

    scenario_warning = [
        RuleScenarioParams(
            "one of two RDMA NICs down",
            {
                RDMA_SYSFS_PRESENT_CMD: CmdOutput(""),
                IBSTAT_CMD: CmdOutput(IBSTAT_ONE_DOWN),
            },
            failed_msg="1/2 RDMA NIC(s) down: mlx5_1/port1; active: mlx5_0/port1 (200 Gbps)",
        ),
    ]

    scenario_not_applicable = [
        RuleScenarioParams(
            "no RDMA subsystem on this node",
            {
                RDMA_SYSFS_PRESENT_CMD: CmdOutput("", return_code=1),
            },
        ),
    ]

    @pytest.mark.parametrize("scenario_params", scenario_passed)
    def test_scenario_passed(self, scenario_params, tested_object):
        RuleTestBase.test_scenario_passed(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_failed)
    def test_scenario_failed(self, scenario_params, tested_object):
        RuleTestBase.test_scenario_failed(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_warning)
    def test_scenario_warning(self, scenario_params, tested_object):
        RuleTestBase.test_scenario_warning(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_passed)
    def test_prerequisite_fulfilled(self, scenario_params, tested_object):
        RuleTestBase.test_prerequisite_fulfilled(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_not_applicable)
    def test_prerequisite_not_fulfilled(self, scenario_params, tested_object):
        RuleTestBase.test_prerequisite_not_fulfilled(self, scenario_params, tested_object)
