#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=protected-access
# pylint: disable=too-many-arguments

"""A module that contains unit-tests for the custom inference model deployment GitHub action."""

import contextlib
import uuid

import pytest
import yaml
from bson import ObjectId
from mock import PropertyMock
from mock import patch

from common import constants
from common.data_types import DataRobotDeployment
from common.data_types import DataRobotModel
from common.exceptions import AssociatedModelNotFound
from common.exceptions import AssociatedModelVersionNotFound
from common.exceptions import DeploymentMetadataAlreadyExists
from common.exceptions import NoValidAncestor
from common.git_tool import GitTool
from common.github_env import GitHubEnv
from common.namepsace import Namespace
from custom_models_action import CustomModelsAction
from deployment_controller import DeploymentController
from deployment_info import DeploymentInfo
from dr_client import DrClient
from model_controller import ModelController
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from schema_validator import SharedSchema
from tests.conftest import unique_str
from tests.unit.conftest import set_namespace
from tests.unit.conftest import validate_metrics
from tests.unit.conftest import validate_namespaced_user_provided_id
from tests.unit.conftest import write_to_file


@pytest.fixture(name="single_deployment_factory")
def fixture_single_deployment_factory(workspace_path, single_model_factory):
    """A factory fixture to create a single deployment along with a model."""

    def _inner(name, write_metadata=True, user_provided_id=None):
        model_name = f"model_{name}"
        model_metadata, _ = single_model_factory(model_name, write_metadata)

        single_deployment_metadata = {
            DeploymentSchema.DEPLOYMENT_ID_KEY: user_provided_id or str(uuid.uuid4()),
            DeploymentSchema.MODEL_ID_KEY: model_metadata[SharedSchema.MODEL_ID_KEY],
        }

        deployment_yaml_filepath = None
        if write_metadata:
            deployment_yaml_filepath = workspace_path / model_name / "deployment.yaml"
            write_to_file(deployment_yaml_filepath, yaml.dump(single_deployment_metadata))

        return single_deployment_metadata, deployment_yaml_filepath

    yield _inner


class TestCustomInferenceDeployment:
    """Contains unit-test for the custom inference deployment GitHub action."""

    @pytest.fixture
    def no_deployments(self, common_path_with_code):
        """A fixture to return a path without deployment definitions in it."""

        yield common_path_with_code

    @contextlib.contextmanager
    def _un_namespaced_deployment_user_provided_id(self, single_or_multi_deployment_metadata):
        if not single_or_multi_deployment_metadata:
            yield
        else:
            if DeploymentSchema.is_multi_deployments_schema(single_or_multi_deployment_metadata):
                if not single_or_multi_deployment_metadata:
                    yield
                else:
                    for deployment_metadata in single_or_multi_deployment_metadata:
                        deployment_metadata[
                            DeploymentSchema.DEPLOYMENT_ID_KEY
                        ] = Namespace.un_namespaced(
                            deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
                        )
                        deployment_metadata[SharedSchema.MODEL_ID_KEY] = Namespace.un_namespaced(
                            deployment_metadata[SharedSchema.MODEL_ID_KEY]
                        )
                    try:
                        yield
                    finally:
                        for deployment_metadata in single_or_multi_deployment_metadata:
                            deployment_metadata[
                                DeploymentSchema.DEPLOYMENT_ID_KEY
                            ] = Namespace.namespaced(
                                deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
                            )
                            deployment_metadata[SharedSchema.MODEL_ID_KEY] = Namespace.namespaced(
                                deployment_metadata[SharedSchema.MODEL_ID_KEY]
                            )
            else:
                single_or_multi_deployment_metadata[
                    DeploymentSchema.DEPLOYMENT_ID_KEY
                ] = Namespace.un_namespaced(
                    single_or_multi_deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
                )
                single_or_multi_deployment_metadata[
                    SharedSchema.MODEL_ID_KEY
                ] = Namespace.un_namespaced(
                    single_or_multi_deployment_metadata[SharedSchema.MODEL_ID_KEY]
                )
                try:
                    yield
                finally:
                    single_or_multi_deployment_metadata[
                        DeploymentSchema.DEPLOYMENT_ID_KEY
                    ] = Namespace.namespaced(
                        single_or_multi_deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
                    )
                    single_or_multi_deployment_metadata[
                        SharedSchema.MODEL_ID_KEY
                    ] = Namespace.namespaced(
                        single_or_multi_deployment_metadata[SharedSchema.MODEL_ID_KEY]
                    )

    # pylint: disable=unused-argument
    @pytest.fixture
    def deployments_factory(self, common_path_with_code, workspace_path, models_factory):
        """A factory fixture to create deployments with associated models."""

        def _inner(num_deployments=2, is_multi=False):
            models_metadata = models_factory(num_deployments, is_multi)

            multi_deployments_metadata = []
            for model_metadata in models_metadata:
                model_name = model_metadata[ModelSchema.SETTINGS_SECTION_KEY][ModelSchema.NAME_KEY]
                deployment_name = f"deployment-{model_name}"
                deployment_metadata = {
                    DeploymentSchema.DEPLOYMENT_ID_KEY: Namespace.namespaced(str(uuid.uuid4())),
                    SharedSchema.MODEL_ID_KEY: model_metadata[SharedSchema.MODEL_ID_KEY],
                    SharedSchema.SETTINGS_SECTION_KEY: {
                        DeploymentSchema.LABEL_KEY: deployment_name
                    },
                }
                multi_deployments_metadata.append(deployment_metadata)
                if not is_multi:
                    deployment_yaml_filepath = workspace_path / model_name / "deployment.yaml"
                    with self._un_namespaced_deployment_user_provided_id(deployment_metadata):
                        write_to_file(deployment_yaml_filepath, yaml.dump(deployment_metadata))

            if is_multi:
                deployments_yaml_filepath = workspace_path / "deployments.yaml"
                with self._un_namespaced_deployment_user_provided_id(multi_deployments_metadata):
                    write_to_file(deployments_yaml_filepath, yaml.dump(multi_deployments_metadata))

            return multi_deployments_metadata, models_metadata

        return _inner

    @pytest.mark.usefixtures("no_deployments")
    def test_scan_and_load_no_deployments(self, options):
        """Test scanning and loading of deployment definitions."""

        deployment_controller = DeploymentController(options, None, None)
        deployment_controller.scan_and_load_deployments_metadata()
        assert len(deployment_controller._deployments_info) == 0

    def test_scan_and_load_deployments_empty_yaml_definition(self, options, single_model_factory):
        """Test deployment' scanning and loading of empty yaml file."""

        name = "model-1"
        _, deployment_yaml_filepath = single_model_factory(name, write_metadata=True)
        with open(deployment_yaml_filepath, "w", encoding="utf-8") as fd:
            fd.write("")
        deployment_controller = DeploymentController(options, None, None)
        deployment_controller.scan_and_load_deployments_metadata()
        assert len(deployment_controller.deployments_info) == 0

    def test_scan_and_load_already_exists_deployment(self, options, single_deployment_factory):
        """Tes scanning and loading of an already existing deployments with same IDs."""

        same_user_provided_id = Namespace.namespaced("123")
        single_deployment_factory("deployment_1", user_provided_id=same_user_provided_id)
        single_deployment_factory("deployment_2", user_provided_id=same_user_provided_id)
        deployment_controller = DeploymentController(options, None, None)
        with pytest.raises(DeploymentMetadataAlreadyExists):
            deployment_controller.scan_and_load_deployments_metadata()

    @pytest.mark.parametrize("namespace", [None, "dev1"], ids=["no-namespace", "dev1-namespace"])
    @pytest.mark.parametrize("num_deployments", [1, 2, 3])
    def test_scan_and_load_deployments_from_multi_separate_yaml_files(
        self, options, single_deployment_factory, namespace, num_deployments
    ):
        """Test scanning and loading of deployments from multi separated yaml files."""

        with set_namespace(namespace):
            for counter in range(1, num_deployments + 1):
                single_deployment_factory(str(counter), write_metadata=True)
            deployment_controller = DeploymentController(options, None, None)
            deployment_controller.scan_and_load_deployments_metadata()
            assert len(deployment_controller._deployments_info) == num_deployments
            validate_namespaced_user_provided_id(
                deployment_controller._deployments_info.values(), namespace
            )

    @pytest.mark.parametrize("namespace", [None, "dev1"], ids=["no-namespace", "dev1-namespace"])
    @pytest.mark.parametrize("num_deployments", [0, 1, 3])
    def test_scan_and_load_deployments_from_one_multi_deployments_yaml_file(
        self, options, deployments_factory, namespace, num_deployments
    ):
        """
        Test scanning and loading of deployments from one multi-deployment definition yaml file.
        """

        with set_namespace(namespace):
            deployments_factory(num_deployments, is_multi=True)
            deployment_controller = DeploymentController(options, None, None)
            deployment_controller.scan_and_load_deployments_metadata()
            assert len(deployment_controller._deployments_info) == num_deployments
            validate_namespaced_user_provided_id(
                deployment_controller._deployments_info.values(), namespace
            )

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
        """Test scanning and loading of models from both single and multi yaml files."""

        deployments_factory(num_multi_deployments, is_multi=True)
        for counter in range(1, num_single_deployments + 1):
            single_deployment_factory(str(counter))

        deployment_controller = DeploymentController(options, None, None)
        deployment_controller.scan_and_load_deployments_metadata()
        assert len(deployment_controller._deployments_info) == (
            num_multi_deployments + num_single_deployments
        )

    @pytest.fixture
    def mock_deployments_metadata(self):
        """A fixture to mock local deployments metadata."""

        return [
            {
                DeploymentSchema.DEPLOYMENT_ID_KEY: Namespace.namespaced("dep-id-1"),
                DeploymentSchema.MODEL_ID_KEY: Namespace.namespaced("model-id-1"),
            },
            {
                DeploymentSchema.DEPLOYMENT_ID_KEY: Namespace.namespaced("dep-id-2"),
                DeploymentSchema.MODEL_ID_KEY: Namespace.namespaced("model-id-2"),
            },
        ]

    def test_deployments_fetching_with_no_associated_dr_model_failure(
        self, options, mock_deployments_metadata
    ):
        """
        Test a failure of deployments fetching from DataRobot with no associated DataRobot
        models.
        """

        with self._mock_datarobot_deployments_with_associated_models(
            mock_deployments_metadata, with_dr_deployments=True, with_associated_dr_models=False
        ):
            model_controller = ModelController(options, None)
            deployment_controller = DeploymentController(options, model_controller, None)
            with pytest.raises(AssociatedModelNotFound):
                deployment_controller.fetch_deployments_from_datarobot()

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
            datarobot_models = self._setup_datarobot_models_map(
                main_branch_sha, with_latest_dr_model_version, deployments_metadata
            )
            datarobot_models_by_model_id = self._setup_datarobot_models_by_model_id(
                datarobot_models
            )

        datarobot_deployments = []
        if with_dr_deployments:
            datarobot_deployments = self._setup_datarobot_deployments(
                deployments_metadata, datarobot_models
            )

        with patch.object(ModelController, "handle_model_changes"), patch.object(
            ModelController,
            "datarobot_models",
            new_callable=PropertyMock(return_value=datarobot_models),
        ), patch.object(
            ModelController,
            "datarobot_model_by_id",
            side_effect=datarobot_models_by_model_id.get,
        ), patch.object(
            DrClient, "fetch_deployments", return_value=datarobot_deployments
        ):
            yield

    @staticmethod
    def _setup_datarobot_models_map(
        main_branch_sha, with_latest_dr_model_version, deployments_metadata
    ):
        datarobot_models = {}
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
            datarobot_models[deployment_metadata[DeploymentSchema.MODEL_ID_KEY]] = DataRobotModel(
                model={"id": datarobot_model_id},
                latest_version=latest_version,
            )
        return datarobot_models

    @staticmethod
    def _setup_datarobot_models_by_model_id(datarobot_models):
        datarobot_models_by_model_id = {}
        for user_provided_id, datarobot_model in datarobot_models.items():
            datarobot_model_id = datarobot_model.model["id"]
            datarobot_models_by_model_id[datarobot_model_id] = datarobot_models[user_provided_id]
        return datarobot_models_by_model_id

    @staticmethod
    def _setup_datarobot_deployments(deployments_metadata, datarobot_models):
        datarobot_deployments = []
        for deployment_metadata in deployments_metadata:
            datarobot_model = datarobot_models.get(
                deployment_metadata[DeploymentSchema.MODEL_ID_KEY]
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
                    "userProvidedId": deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY],
                    "model": {
                        "customModelImage": {
                            "customModelId": custom_model_id,
                            "customModelVersionId": custom_model_ver_id,
                        }
                    },
                }
            )
        return datarobot_deployments

    def test_deployments_fetching_with_no_dr_latest_model_version_failure(
        self, options, mock_deployments_metadata
    ):
        """Test a failure of deployment fetching with no associated latest model version."""

        with self._mock_datarobot_deployments_with_associated_models(
            mock_deployments_metadata,
            with_dr_deployments=True,
            with_associated_dr_models=True,
            with_latest_dr_model_version=False,
        ):
            model_controller = ModelController(options, None)
            deployment_controller = DeploymentController(options, model_controller, None)
            with pytest.raises(AssociatedModelVersionNotFound):
                deployment_controller.fetch_deployments_from_datarobot()

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

    @pytest.mark.parametrize(
        "namespace", [None, "dev1"], ids=["default-namespace", "dev1-namespace"]
    )
    @pytest.mark.usefixtures("workspace_path")
    def test_deployments_integrity_validation_success(
        self, options, deployments_factory, git_repo, init_repo_for_root_path_factory, namespace
    ):
        """Test a successful deployments' integrity validation."""

        with set_namespace(namespace):
            with self._mock_repo_with_datarobot_models(
                deployments_factory, git_repo, init_repo_for_root_path_factory
            ):
                self._setup_and_run_deployment_integrity_validation(options)

    @staticmethod
    def _setup_and_run_deployment_integrity_validation(options):
        git_tool = GitTool(GitHubEnv.workspace_path())
        model_controller = ModelController(options, git_tool)
        model_controller.scan_and_load_models_metadata()
        deployment_controller = DeploymentController(options, model_controller, git_tool)
        deployment_controller.scan_and_load_deployments_metadata()
        deployment_controller.fetch_deployments_from_datarobot()
        deployment_controller.validate_deployments_integrity()

    @pytest.mark.usefixtures("workspace_path")
    def test_deployments_integrity_validation_no_dr_deployments(
        self, options, deployments_factory, git_repo, init_repo_for_root_path_factory
    ):
        """
        Test a successful deployments' integrity validation with no associated DataRobot
        deployments.
        """
        with self._mock_repo_with_datarobot_models(
            deployments_factory,
            git_repo,
            init_repo_for_root_path_factory,
            with_dr_deployments=False,
        ):
            self._setup_and_run_deployment_integrity_validation(options)

    @pytest.mark.usefixtures("no_deployments")
    def test_save_statistics(self, options, github_output):
        """A case to test deployment's statistics saving."""

        with patch("dr_client.DrClient"):
            deployment_controller = DeploymentController(options, None, None)
            validate_metrics(github_output, constants.Label.DEPLOYMENTS, deployment_controller)

    @pytest.mark.usefixtures("workspace_path", "mock_github_env_variables")
    def test_deployments_integrity_validation_no_associated_models(
        self, options, deployments_factory, git_repo, init_repo_for_root_path_factory
    ):
        """
        Test a failure of deployments integrity validation with no associated models.
        """

        with self._mock_repo_with_datarobot_models(
            deployments_factory,
            git_repo,
            init_repo_for_root_path_factory,
            with_dr_deployments=False,
            with_associated_dr_models=False,
        ):
            custom_inference_deployment = CustomModelsAction(options)
            with patch.object(CustomModelsAction, "_save_statistics"), patch.object(
                ModelController, "fetch_models_from_datarobot"
            ):
                with pytest.raises(AssociatedModelNotFound):
                    custom_inference_deployment.run()

    @pytest.mark.usefixtures("workspace_path")
    def test_deployments_integrity_validation_no_latest_version(
        self, options, deployments_factory, git_repo, init_repo_for_root_path_factory
    ):
        """Test a failure of deployments integrity validation with no latest model version."""

        with self._mock_repo_with_datarobot_models(
            deployments_factory,
            git_repo,
            init_repo_for_root_path_factory,
            with_dr_deployments=False,
            with_associated_dr_models=True,
            with_latest_dr_model_version=False,
        ):
            with pytest.raises(AssociatedModelVersionNotFound):
                self._setup_and_run_deployment_integrity_validation(options)

    @pytest.mark.usefixtures("workspace_path")
    def test_deployments_integrity_validation_no_main_branch_sha_failure(
        self, options, deployments_factory, git_repo, init_repo_for_root_path_factory
    ):
        """Test a failure of deployments integrity validation with no expected main branch SHA."""

        with self._mock_repo_with_datarobot_models(
            deployments_factory,
            git_repo,
            init_repo_for_root_path_factory,
            with_main_branch_sha=False,
        ):
            with pytest.raises(NoValidAncestor):
                self._setup_and_run_deployment_integrity_validation(options)


class TestDeploymentChanges:
    """Contains cases to test user's changes in a local deployment definition."""

    @pytest.fixture
    def _mock_datarobot_model_factory(self):
        """A fixture to create a datarobot model class."""

        def _inner(model_id=None, user_provided_id=None, latest_version_id=None):
            model_id = model_id or unique_str()
            user_provided_id = user_provided_id or unique_str()
            latest_version_id = latest_version_id or unique_str()
            return DataRobotModel(
                model={"id": model_id, "userProvidedId": user_provided_id},
                latest_version={"id": latest_version_id, "customModelId": model_id},
            )

        return _inner

    @pytest.fixture
    def _mock_datarobot_deployment_factory(self, _mock_datarobot_model_factory):
        """A fixture to create a datarobot deployment class."""

        def _inner(
            model_id=None,
            user_provided_model_id=None,
            latest_version_id=None,
            user_provided_deployment_id=None,
        ):
            datarobot_model = _mock_datarobot_model_factory(
                model_id, user_provided_model_id, latest_version_id
            )
            user_provided_deployment_id = user_provided_deployment_id or unique_str()
            return DataRobotDeployment(
                deployment={"id": unique_str(), "userProvidedId": user_provided_deployment_id},
                model_version=datarobot_model.latest_version,
            )

        return _inner

    @pytest.fixture
    def _mock_deployment_info_factory(self):
        def _inner(user_provided_id, user_provided_model_id=None, enable_challenger=True):
            return DeploymentInfo(
                yaml_path="/dummy/path.yaml",
                deployment_metadata={
                    DeploymentSchema.DEPLOYMENT_ID_KEY: user_provided_id,
                    DeploymentSchema.MODEL_ID_KEY: user_provided_model_id or unique_str(),
                    DeploymentSchema.SETTINGS_SECTION_KEY: {
                        DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY: enable_challenger
                    },
                },
            )

        return _inner

    def test_new_deployment_creation(
        self, options, _mock_deployment_info_factory, _mock_datarobot_deployment_factory
    ):
        """A case to test new deployment creation."""

        one_user_provided_id = unique_str()
        another_user_provided_id = unique_str()
        deployment_info = _mock_deployment_info_factory(one_user_provided_id)
        datarobot_deployment = _mock_datarobot_deployment_factory(another_user_provided_id)

        with patch.object(
            DeploymentController,
            "deployments_info",
            new_callable=PropertyMock(return_value={one_user_provided_id: deployment_info}),
        ), patch.object(
            DeploymentController,
            "datarobot_deployments",
            new_callable=PropertyMock(
                return_value={another_user_provided_id: datarobot_deployment}
            ),
        ), patch.object(
            DeploymentController, "_create_deployment"
        ) as create_deployment_method:
            custom_inference_deployment = DeploymentController(options, None, None)
            custom_inference_deployment.handle_deployment_changes_or_creation()

            create_deployment_method.assert_called_once()

    @pytest.fixture
    def _patch_handle_deployment_settings(self):
        with patch.object(DeploymentController, "_handle_deployment_settings"):
            yield

    @pytest.mark.usefixtures("_patch_handle_deployment_settings")
    def test_model_id_change_in_existing_deployment(
        self, options, _mock_datarobot_model_factory, _mock_deployment_info_factory
    ):
        """A case to test a model ID change in existing active deployment."""

        origin_model_id = unique_str()
        origin_datarobot_model = _mock_datarobot_model_factory(origin_model_id)
        new_model_id = unique_str()
        new_datarobot_model = _mock_datarobot_model_factory(new_model_id)

        user_provided_id = unique_str()
        datarobot_deployment = DataRobotDeployment(
            deployment={"id": unique_str(), "userProvidedId": user_provided_id},
            model_version=origin_datarobot_model.latest_version,
        )
        deployment_info = _mock_deployment_info_factory(
            user_provided_id, new_model_id, enable_challenger=True
        )

        with patch.object(
            DeploymentController,
            "deployments_info",
            new_callable=PropertyMock(return_value={user_provided_id: deployment_info}),
        ), patch.object(
            DeploymentController,
            "datarobot_deployments",
            new_callable=PropertyMock(return_value={user_provided_id: datarobot_deployment}),
        ), patch.object(
            ModelController,
            "datarobot_models",
            new_callable=PropertyMock(return_value={new_model_id: new_datarobot_model}),
        ), patch.object(
            DeploymentController, "_create_challenger_in_deployment_if_not_created_already"
        ) as create_challenger_in_deployment_method:
            model_controller = ModelController(options, None)
            custom_inference_deployment = DeploymentController(options, model_controller, None)
            custom_inference_deployment.handle_deployment_changes_or_creation()

            create_challenger_in_deployment_method.assert_called_once()

    @pytest.mark.usefixtures("_patch_handle_deployment_settings")
    def test_new_model_version_in_existing_deployment(
        self, options, _mock_datarobot_model_factory, _mock_deployment_info_factory
    ):
        """A case to test a new model version for a model in existing active deployment."""

        model_id = unique_str()
        origin_latest_version_id = unique_str()
        datarobot_model_with_origin_latest = _mock_datarobot_model_factory(
            model_id, latest_version_id=origin_latest_version_id
        )
        new_latest_version_id = unique_str()
        datarobot_model_with_new_latest = _mock_datarobot_model_factory(
            model_id, latest_version_id=new_latest_version_id
        )
        datarobot_deployment = DataRobotDeployment(
            deployment={"id": unique_str(), "userProvidedId": unique_str()},
            model_version=datarobot_model_with_origin_latest.latest_version,
        )
        user_provided_id = unique_str()
        deployment_info = _mock_deployment_info_factory(
            user_provided_id, model_id, enable_challenger=True
        )

        with patch.object(
            DeploymentController,
            "deployments_info",
            new_callable=PropertyMock(return_value={user_provided_id: deployment_info}),
        ), patch.object(
            DeploymentController,
            "datarobot_deployments",
            new_callable=PropertyMock(return_value={user_provided_id: datarobot_deployment}),
        ), patch.object(
            ModelController,
            "datarobot_models",
            new_callable=PropertyMock(return_value={model_id: datarobot_model_with_new_latest}),
        ), patch.object(
            DeploymentController, "_create_challenger_in_deployment_if_not_created_already"
        ) as create_challenger_in_deployment_method:
            model_controller = ModelController(options, None)
            deployment_controller = DeploymentController(options, model_controller, None)
            deployment_controller.handle_deployment_changes_or_creation()

            create_challenger_in_deployment_method.assert_called_once()
