"""Tests for NVIDIA GPU hardware validations."""

import pytest

from in_cluster_checks.rules.gpu.gpu_validations import (
    VerifyGpuDriverInstalled,
    VerifyGpuEccErrorsAbsent,
)
from tests.pytest_tools.test_operator_base import CmdOutput
from tests.pytest_tools.test_rule_base import RuleScenarioParams, RuleTestBase

WHICH_NVIDIA_SMI_CMD = "which nvidia-smi"


class TestVerifyGpuDriverInstalled(RuleTestBase):
    """Test VerifyGpuDriverInstalled rule."""

    tested_type = VerifyGpuDriverInstalled

    driver_cmd = str(VerifyGpuDriverInstalled.DRIVER_QUERY_CMD)

    single_gpu_output = "580.126.09, NVIDIA A100 80GB PCIe, 81920"
    multi_gpu_output = "580.126.09, NVIDIA H100 80GB HBM3, 81559\n580.126.09, NVIDIA H100 80GB HBM3, 81559"

    scenario_passed = [
        RuleScenarioParams(
            "single GPU reporting driver info",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("/usr/bin/nvidia-smi"),
                driver_cmd: CmdOutput(single_gpu_output),
            },
        ),
        RuleScenarioParams(
            "multiple GPUs reporting driver info",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("/usr/bin/nvidia-smi"),
                driver_cmd: CmdOutput(multi_gpu_output),
            },
        ),
    ]

    scenario_failed = [
        RuleScenarioParams(
            "nvidia-smi command fails",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("/usr/bin/nvidia-smi"),
                driver_cmd: CmdOutput("", return_code=1, err="NVIDIA-SMI has failed"),
            },
            failed_msg="nvidia-smi failed: NVIDIA-SMI has failed",
        ),
    ]

    scenario_not_applicable = [
        RuleScenarioParams(
            "nvidia-smi not available on node",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("", return_code=1),
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


class TestVerifyGpuEccErrorsAbsent(RuleTestBase):
    """Test VerifyGpuEccErrorsAbsent rule."""

    tested_type = VerifyGpuEccErrorsAbsent

    ecc_cmd = str(VerifyGpuEccErrorsAbsent.ECC_QUERY_CMD)

    scenario_passed = [
        RuleScenarioParams(
            "no ECC errors on any GPU",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("/usr/bin/nvidia-smi"),
                ecc_cmd: CmdOutput("0, 0\n1, 0"),
            },
        ),
        RuleScenarioParams(
            "ECC reporting not supported (N/A)",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("/usr/bin/nvidia-smi"),
                ecc_cmd: CmdOutput("0, N/A"),
            },
        ),
    ]

    scenario_failed = [
        RuleScenarioParams(
            "uncorrectable ECC errors found on one GPU",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("/usr/bin/nvidia-smi"),
                ecc_cmd: CmdOutput("0, 0\n1, 3"),
            },
            failed_msg=(
                "Uncorrectable ECC errors found: GPU 1: 3 uncorrectable errors. "
                "Replace GPU or contact cloud provider"
            ),
        ),
    ]

    scenario_warning = [
        RuleScenarioParams(
            "nvidia-smi ECC query command fails",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("/usr/bin/nvidia-smi"),
                ecc_cmd: CmdOutput("", return_code=1, err="ECC query not supported"),
            },
            failed_msg="nvidia-smi ECC query failed: ECC query not supported",
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
