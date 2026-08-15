"""Tests for NVIDIA GPU hardware validations."""

from unittest.mock import Mock

import pytest

from in_cluster_checks.rules.gpu.gpu_validations import (
    VerifyGpuDriverInstalled,
    VerifyGpuEccErrorsAbsent,
    VerifyGpuNodeLabelPresent,
)
from tests.pytest_tools.test_operator_base import CmdOutput
from tests.pytest_tools.test_rule_base import RuleScenarioParams, RuleTestBase

WHICH_NVIDIA_SMI_CMD = "which nvidia-smi"


def _fake_node(labels: dict | None = None, allocatable: dict | None = None) -> Mock:
    """Build a fake openshift_client node object with the given labels/allocatable resources."""
    node = Mock()
    node.model.metadata.labels = labels or {}
    node.model.status.allocatable = allocatable or {}
    return node


_GPU_LABEL_NODE = _fake_node(labels={"nvidia.com/gpu.present": "true"})
_GPU_RESOURCE_NODE = _fake_node(allocatable={"nvidia.com/gpu": "4"})
_NON_GPU_NODE = _fake_node(labels={"kubernetes.io/hostname": "worker-0"}, allocatable={"cpu": "16"})


class TestVerifyGpuNodeLabelPresent(RuleTestBase):
    """Test VerifyGpuNodeLabelPresent rule."""

    tested_type = VerifyGpuNodeLabelPresent

    scenario_passed = [
        RuleScenarioParams(
            "node has the nvidia.com/gpu.present label",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=_GPU_LABEL_NODE)},
        ),
    ]

    scenario_failed = [
        RuleScenarioParams(
            "node could not be found via the cluster API",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=None)},
            failed_msg="Node test-node could not be found via the cluster API",
        ),
    ]

    scenario_warning = [
        RuleScenarioParams(
            "node has schedulable GPU resources but is missing the label",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=_GPU_RESOURCE_NODE)},
            failed_msg=(
                "Node has 4 schedulable nvidia.com/gpu resource(s) but is missing "
                "the nvidia.com/gpu.present label"
            ),
        ),
    ]

    scenario_not_applicable = [
        RuleScenarioParams(
            "node has no GPU labels/resources",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=_NON_GPU_NODE)},
            failed_msg="Node test-node has no GPU labels/resources (not a GPU node)",
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

    @pytest.mark.parametrize("scenario_params", scenario_not_applicable)
    def test_scenario_not_applicable(self, scenario_params, tested_object):
        RuleTestBase.test_scenario_not_applicable(self, scenario_params, tested_object)


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

    scenario_warning = [
        RuleScenarioParams(
            "nvidia-smi binary not available on this GPU node",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("", return_code=1),
            },
            failed_msg=VerifyGpuDriverInstalled.NVIDIA_SMI_NOT_AVAILABLE_MSG,
        ),
    ]

    scenario_prerequisite_met = [
        RuleScenarioParams(
            "node has the nvidia.com/gpu.present label",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=_GPU_LABEL_NODE)},
        ),
        RuleScenarioParams(
            "node has the nvidia.com/gpu allocatable resource",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=_GPU_RESOURCE_NODE)},
        ),
    ]

    scenario_prerequisite_not_met = [
        RuleScenarioParams(
            "node has no GPU labels/resources",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=_NON_GPU_NODE)},
        ),
        RuleScenarioParams(
            "node could not be found via the cluster API",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=None)},
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

    @pytest.mark.parametrize("scenario_params", scenario_prerequisite_met)
    def test_prerequisite_fulfilled(self, scenario_params, tested_object):
        RuleTestBase.test_prerequisite_fulfilled(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_prerequisite_not_met)
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
        RuleScenarioParams(
            "nvidia-smi binary not available on this GPU node",
            {
                WHICH_NVIDIA_SMI_CMD: CmdOutput("", return_code=1),
            },
            failed_msg=VerifyGpuEccErrorsAbsent.NVIDIA_SMI_NOT_AVAILABLE_MSG,
        ),
    ]

    scenario_prerequisite_met = [
        RuleScenarioParams(
            "node has the nvidia.com/gpu.present label",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=_GPU_LABEL_NODE)},
        ),
    ]

    scenario_prerequisite_not_met = [
        RuleScenarioParams(
            "node has no GPU labels/resources",
            tested_object_mock_dict={"oc_api.select_resources": Mock(return_value=_NON_GPU_NODE)},
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

    @pytest.mark.parametrize("scenario_params", scenario_prerequisite_met)
    def test_prerequisite_fulfilled(self, scenario_params, tested_object):
        RuleTestBase.test_prerequisite_fulfilled(self, scenario_params, tested_object)

    @pytest.mark.parametrize("scenario_params", scenario_prerequisite_not_met)
    def test_prerequisite_not_fulfilled(self, scenario_params, tested_object):
        RuleTestBase.test_prerequisite_not_fulfilled(self, scenario_params, tested_object)
