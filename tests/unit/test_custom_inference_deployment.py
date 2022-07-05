import json
import uuid

import pytest
import yaml

from common.exceptions import DeploymentMetadataAlreadyExists
from custom_inference_deployment import CustomInferenceDeployment
from schema_validator import DeploymentSchema, ModelSchema, SharedSchema
from tests.unit.conftest import write_to_file


class TestCustomInferenceDeployment:
    @pytest.fixture
    def no_deployments(self, common_path_with_code):
        yield common_path_with_code

    @pytest.fixture
    def single_deployment_factory(
        self, repo_root_path, common_path_with_code, excluded_src_path, single_model_factory
    ):
        def _inner(name, write_metadata=True, git_deployment_id=None, git_model_id=None):
            model_name = f"model_{name}"
            model_metadata = single_model_factory(
                model_name, write_metadata, git_model_id=git_model_id
            )

            single_deployment_metadata = {
                DeploymentSchema.DEPLOYMENT_ID_KEY: (
                    git_deployment_id if git_deployment_id else str(uuid.uuid4())
                ),
                DeploymentSchema.MODEL_ID_KEY: model_metadata[SharedSchema.MODEL_ID_KEY],
            }

            if write_metadata:
                deployment_yaml_filepath = repo_root_path / model_name / "deployment.yaml"
                write_to_file(deployment_yaml_filepath, yaml.dump(single_deployment_metadata))

            return single_deployment_metadata

        yield _inner

    @pytest.fixture
    def deployments_factory(self, repo_root_path, common_path_with_code, models_factory):
        def _inner(num_deployments=2, is_multi=False):
            models_metadata = models_factory(num_deployments, is_multi)

            multi_deployments_metadata = []
            for model_metadata in models_metadata:
                model_name = model_metadata[ModelSchema.SETTINGS_SECTION_KEY][ModelSchema.NAME_KEY]
                deployment_name = f"deployment-{model_name}"
                deployment_metadata = {
                    DeploymentSchema.DEPLOYMENT_ID_KEY: str(uuid.uuid4()),
                    SharedSchema.MODEL_ID_KEY: model_metadata[SharedSchema.MODEL_ID_KEY],
                    SharedSchema.SETTINGS_SECTION_KEY: {
                        DeploymentSchema.LABEL_KEY: deployment_name
                    },
                }
                multi_deployments_metadata.append(deployment_metadata)
                if not is_multi:
                    deployment_yaml_filepath = repo_root_path / model_name / "deployment.yaml"
                    write_to_file(deployment_yaml_filepath, json.dump(deployment_metadata))

            if is_multi:
                deployments_yaml_filepath = repo_root_path / "deployments.yaml"
                write_to_file(deployments_yaml_filepath, yaml.dump(multi_deployments_metadata))

            return deployments_yaml_filepath

        return _inner

    @pytest.mark.usefixtures("no_deployments")
    def test_scan_and_load_no_deployments(self, options):
        custom_inference_deployment = CustomInferenceDeployment(options)
        custom_inference_deployment._scan_and_load_deployments_metadata()
        assert len(custom_inference_deployment._deployments_info) == 0

    def test_scan_and_load_already_exists_deployment(self, options, single_deployment_factory):
        same_git_deployment_id = "123"
        single_deployment_factory("deployment_1", git_deployment_id=same_git_deployment_id)
        single_deployment_factory("deployment_2", git_deployment_id=same_git_deployment_id)
        custom_inference_deployment = CustomInferenceDeployment(options)
        with pytest.raises(DeploymentMetadataAlreadyExists):
            custom_inference_deployment._scan_and_load_deployments_metadata()

    @pytest.mark.parametrize("num_deployments", [1, 2, 3])
    def test_scan_and_load_deployments_from_multi_separate_yaml_files(
        self, options, single_deployment_factory, num_deployments
    ):
        for counter in range(1, num_deployments + 1):
            single_deployment_factory(str(counter), write_metadata=True)
        custom_inference_deployment = CustomInferenceDeployment(options)
        custom_inference_deployment._scan_and_load_deployments_metadata()
        assert len(custom_inference_deployment._deployments_info) == num_deployments

    @pytest.mark.parametrize("num_deployments", [0, 1, 3])
    def test_scan_and_load_deployments_from_one_multi_deployments_yaml_file(
        self, options, deployments_factory, num_deployments
    ):
        deployments_factory(num_deployments, is_multi=True)
        custom_inference_deployment = CustomInferenceDeployment(options)
        custom_inference_deployment._scan_and_load_deployments_metadata()
        assert len(custom_inference_deployment._deployments_info) == num_deployments

    @pytest.mark.parametrize("num_single_deployments", [0, 1, 3])
    @pytest.mark.parametrize("num_multi_deployments", [0, 2])
    def test_scan_and_load_models_from_both_multi_and_single_yaml_files(
        self,
        options,
        num_single_deployments,
        num_multi_deployments,
        single_deployment_factory,
        deployments_factory,
    ):
        deployments_factory(num_multi_deployments, is_multi=True)
        for counter in range(1, num_single_deployments + 1):
            single_deployment_factory(str(counter))

        custom_inference_deployment = CustomInferenceDeployment(options)
        custom_inference_deployment._scan_and_load_deployments_metadata()
        assert len(custom_inference_deployment._deployments_info) == (
            num_multi_deployments + num_single_deployments
        )
