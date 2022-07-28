import contextlib
import uuid

import pytest
import yaml
from bson import ObjectId
from mock import patch
from mock import PropertyMock

from common.data_types import DataRobotModel
from common.exceptions import AssociatedModelNotFound
from common.exceptions import AssociatedModelVersionNotFound
from common.exceptions import DeploymentMetadataAlreadyExists
from common.exceptions import NoValidAncestor
from custom_inference_deployment import CustomInferenceDeployment
from custom_inference_model import CustomInferenceModelBase
from dr_client import DrClient
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
                    write_to_file(deployment_yaml_filepath, yaml.dump(deployment_metadata))

            if is_multi:
                deployments_yaml_filepath = repo_root_path / "deployments.yaml"
                write_to_file(deployments_yaml_filepath, yaml.dump(multi_deployments_metadata))

            return multi_deployments_metadata, models_metadata

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

    @pytest.fixture
    def mock_datarobot_deployments(self):
        return [
            {"git_datarobot_deployment_id": "dep-id-1", "git_datarobot_model_id": "model-id-1"},
            {"git_datarobot_deployment_id": "dep-id-2", "git_datarobot_model_id": "model-id-2"},
        ]

    def test_deployments_fetching_with_no_associated_dr_model_failure(
        self, options, mock_datarobot_deployments
    ):
        with self._mock_datarobot_deployments_with_associated_models(
            mock_datarobot_deployments, with_dr_deployments=True, with_associated_dr_models=False
        ):
            custom_inference_deployment = CustomInferenceDeployment(options)
            with pytest.raises(AssociatedModelNotFound):
                custom_inference_deployment._fetch_deployments_from_datarobot()

    @contextlib.contextmanager
    def _mock_datarobot_deployments_with_associated_models(
        self,
        deployments_metadata,
        main_branch_sha=None,
        with_dr_deployments=True,
        with_associated_dr_models=True,
        with_latest_dr_model_version=True,
    ):
        datarobot_models = {}
        datarobot_models_by_model_id = {}

        if with_associated_dr_models:
            for deployment_metadata in deployments_metadata:
                datarobot_model_id = str(ObjectId())
                datarobot_model_version_id = str(ObjectId())
                latest_version = (
                    {
                        "id": datarobot_model_version_id,
                        "gitModelVersion": {"mainBranchCommitSha": main_branch_sha},
                    }
                    if with_latest_dr_model_version
                    else None
                )
                datarobot_models[deployment_metadata["git_datarobot_model_id"]] = DataRobotModel(
                    model={"id": datarobot_model_id},
                    latest_version=latest_version,
                )
                datarobot_models_by_model_id[datarobot_model_id] = datarobot_models[
                    deployment_metadata["git_datarobot_model_id"]
                ]

        datarobot_deployments = []
        if with_dr_deployments:
            for deployment_metadata in deployments_metadata:
                datarobot_model = datarobot_models.get(
                    deployment_metadata["git_datarobot_model_id"]
                )
                # DataRobot Deployments
                custom_model_id = datarobot_model.model["id"] if datarobot_model else None
                custom_model_ver_id = (
                    datarobot_model.latest_version["id"]
                    if datarobot_model and datarobot_model.latest_version
                    else None
                )
                datarobot_deployments.append(
                    {
                        "gitDeploymentId": deployment_metadata["git_datarobot_deployment_id"],
                        "model": {
                            "customModelImage": {
                                "customModelId": custom_model_id,
                                "customModelVersionId": custom_model_ver_id,
                            }
                        },
                    }
                )

        with patch.object(
            CustomInferenceModelBase,
            "datarobot_models",
            new_callable=PropertyMock(return_value=datarobot_models),
        ), patch.object(
            CustomInferenceModelBase,
            "datarobot_model_by_id",
            side_effect=lambda model_id: datarobot_models_by_model_id.get(model_id),
        ), patch.object(
            DrClient, "fetch_deployments", return_value=datarobot_deployments
        ):
            yield

    def test_deployments_fetching_with_no_dr_latest_model_version_failure(
        self, options, mock_datarobot_deployments
    ):
        with self._mock_datarobot_deployments_with_associated_models(
            mock_datarobot_deployments,
            with_dr_deployments=True,
            with_associated_dr_models=True,
            with_latest_dr_model_version=False,
        ):
            custom_inference_deployment = CustomInferenceDeployment(options)
            with pytest.raises(AssociatedModelVersionNotFound):
                custom_inference_deployment._fetch_deployments_from_datarobot()

    @contextlib.contextmanager
    def _mock_repo_with_datarobot_models(
        self,
        deployments_factory,
        git_repo,
        init_repo_for_root_path_factory,
        with_dr_deployments=True,
        with_associated_dr_models=True,
        with_latest_dr_model_version=True,
        with_main_branch_sha=True,
    ):
        deployments_metadata, _ = deployments_factory()
        init_repo_for_root_path_factory()

        main_branch_sha = git_repo.head.commit.hexsha if with_main_branch_sha else None
        with self._mock_datarobot_deployments_with_associated_models(
            deployments_metadata,
            main_branch_sha,
            with_dr_deployments,
            with_associated_dr_models,
            with_latest_dr_model_version,
        ):
            yield

    def test_deployments_integrity_validation_success(
        self,
        options,
        repo_root_path,
        deployments_factory,
        git_repo,
        init_repo_for_root_path_factory,
    ):
        with self._mock_repo_with_datarobot_models(
            deployments_factory, git_repo, init_repo_for_root_path_factory
        ):
            custom_inference_deployment = CustomInferenceDeployment(options)
            custom_inference_deployment._scan_and_load_deployments_metadata()
            custom_inference_deployment._fetch_deployments_from_datarobot()
            custom_inference_deployment._validate_deployments_integrity()

    def test_deployments_integrity_validation_no_dr_deployments(
        self,
        options,
        repo_root_path,
        deployments_factory,
        git_repo,
        init_repo_for_root_path_factory,
    ):
        with self._mock_repo_with_datarobot_models(
            deployments_factory,
            git_repo,
            init_repo_for_root_path_factory,
            with_dr_deployments=False,
        ):
            custom_inference_deployment = CustomInferenceDeployment(options)
            custom_inference_deployment._scan_and_load_models_metadata()
            custom_inference_deployment._scan_and_load_deployments_metadata()
            custom_inference_deployment._fetch_deployments_from_datarobot()
            custom_inference_deployment._validate_deployments_integrity()

    def test_deployments_integrity_validation_no_associated_models(
        self,
        options,
        repo_root_path,
        deployments_factory,
        git_repo,
        init_repo_for_root_path_factory,
    ):
        with self._mock_repo_with_datarobot_models(
            deployments_factory,
            git_repo,
            init_repo_for_root_path_factory,
            with_dr_deployments=False,
            with_associated_dr_models=False,
        ):
            custom_inference_deployment = CustomInferenceDeployment(options)
            custom_inference_deployment._scan_and_load_models_metadata()
            custom_inference_deployment._scan_and_load_deployments_metadata()
            custom_inference_deployment._fetch_deployments_from_datarobot()
            with pytest.raises(AssociatedModelNotFound):
                custom_inference_deployment._validate_deployments_integrity()

    def test_deployments_integrity_validation_no_latest_version(
        self,
        options,
        repo_root_path,
        deployments_factory,
        git_repo,
        init_repo_for_root_path_factory,
    ):
        with self._mock_repo_with_datarobot_models(
            deployments_factory,
            git_repo,
            init_repo_for_root_path_factory,
            with_dr_deployments=False,
            with_associated_dr_models=True,
            with_latest_dr_model_version=False,
        ):
            custom_inference_deployment = CustomInferenceDeployment(options)
            custom_inference_deployment._scan_and_load_models_metadata()
            custom_inference_deployment._scan_and_load_deployments_metadata()
            custom_inference_deployment._fetch_deployments_from_datarobot()
            with pytest.raises(AssociatedModelVersionNotFound):
                custom_inference_deployment._validate_deployments_integrity()

    def test_deployments_integrity_validation_no_main_branch_sha_failure(
        self,
        options,
        repo_root_path,
        deployments_factory,
        git_repo,
        init_repo_for_root_path_factory,
    ):
        with self._mock_repo_with_datarobot_models(
            deployments_factory,
            git_repo,
            init_repo_for_root_path_factory,
            with_main_branch_sha=False,
        ):
            custom_inference_deployment = CustomInferenceDeployment(options)
            custom_inference_deployment._scan_and_load_deployments_metadata()
            custom_inference_deployment._fetch_deployments_from_datarobot()
            with pytest.raises(NoValidAncestor):
                custom_inference_deployment._validate_deployments_integrity()
