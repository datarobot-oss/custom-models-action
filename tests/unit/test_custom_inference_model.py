#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=protected-access
# pylint: disable=too-many-arguments

"""A module that contains unit-tests for the custom inference model GitHub action."""
import argparse
import contextlib
import os
from pathlib import Path

import pytest
from bson import ObjectId
from mock import patch
from mock.mock import PropertyMock

from common import constants
from common.data_types import DataRobotModel
from common.exceptions import IllegalModelDeletion
from common.exceptions import ModelMainEntryPointNotFound
from common.exceptions import ModelMetadataAlreadyExists
from common.exceptions import SharedAndLocalPathCollision
from common.git_tool import GitTool
from common.github_env import GitHubEnv
from custom_models_action import CustomModelsAction
from deployment_controller import DeploymentController
from dr_client import DrClient
from model_controller import ModelController
from model_file_path import ModelFilePath
from model_info import ModelInfo
from tests.unit.conftest import make_a_change_and_commit
from tests.unit.conftest import set_namespace
from tests.unit.conftest import validate_metrics
from tests.unit.conftest import validate_namespaced_user_provided_id


class TestCustomInferenceModel:
    """Contains unit-tests for the custom inference model GitHub action."""

    @pytest.fixture
    def no_models(self, common_path_with_code):
        """A fixture to return a path with not model definitions in it."""

        yield common_path_with_code

    @pytest.mark.usefixtures("no_models")
    def test_scan_and_load_no_models(self, options):
        """Test models' scanning and loading without existing model definitions."""

        model_controller = ModelController(options, None)
        model_controller.scan_and_load_models_metadata()
        assert len(model_controller.models_info) == 0

    def test_scan_and_load_models_empty_yaml_definition(self, options, single_model_factory):
        """Test models' scanning and loading of empty yaml file."""

        name = "model-1"
        _, model_yaml_filepath = single_model_factory(name, write_metadata=True)
        with open(model_yaml_filepath, "w", encoding="utf-8") as fd:
            fd.write("")
        model_controller = ModelController(options, None)
        model_controller.scan_and_load_models_metadata()
        assert len(model_controller.models_info) == 0

    @pytest.mark.parametrize("namespace", [None, "dev1"], ids=["no-namespace", "dev1-namespace"])
    @pytest.mark.parametrize("num_models", [1, 2, 3])
    def test_scan_and_load_models_from_multi_separate_yaml_files(
        self, options, single_model_factory, namespace, num_models
    ):
        """Test models' scanning and loading from multiple separated yaml files."""

        with set_namespace(namespace):
            for counter in range(1, num_models + 1):
                single_model_factory(f"model-{counter}", write_metadata=True)
            model_controller = ModelController(options, None)
            model_controller.scan_and_load_models_metadata()
            assert len(model_controller.models_info) == num_models
            validate_namespaced_user_provided_id(model_controller.models_info.values(), namespace)

    def test_scan_and_load_models_with_same_user_provided_id_failure(
        self, options, single_model_factory
    ):
        """Test a failure of models' scanning and loading of multiple models with same ID."""

        user_provided_id = "same-user-provided-id-111"
        single_model_factory("model-1", write_metadata=True, user_provided_id=user_provided_id)
        single_model_factory("model-2", write_metadata=True, user_provided_id=user_provided_id)
        model_controller = ModelController(options, None)
        with pytest.raises(ModelMetadataAlreadyExists):
            model_controller.scan_and_load_models_metadata()

    @pytest.mark.parametrize("namespace", [None, "dev1"], ids=["no-namespace", "dev1-namespace"])
    @pytest.mark.parametrize("num_models", [0, 1, 3])
    @pytest.mark.parametrize(
        "is_absolute_path, root_prefix", [(True, "/"), (True, "$ROOT/"), (False, None)]
    )
    def test_scan_and_load_models_from_one_multi_models_yaml_file(
        self, options, models_factory, namespace, num_models, is_absolute_path, root_prefix
    ):
        """Test models' scanning and load from a single multi-models yaml definition."""

        with set_namespace(namespace):
            models_factory(
                num_models,
                is_multi=True,
                is_absolute_path=is_absolute_path,
                root_prefix=root_prefix,
            )
            model_controller = ModelController(options, None)
            model_controller.scan_and_load_models_metadata()
            assert len(model_controller.models_info) == num_models
            for model_info in model_controller.models_info.values():
                assert model_info.model_path.is_dir()
            validate_namespaced_user_provided_id(model_controller.models_info.values(), namespace)

    @pytest.mark.parametrize("num_single_models", [0, 1, 3])
    @pytest.mark.parametrize("num_multi_models", [0, 2])
    def test_scan_and_load_models_from_both_multi_and_single_yaml_files(
        self,
        options,
        num_single_models,
        num_multi_models,
        single_model_factory,
        models_factory,
    ):
        """Test models' scanning and loading from both multi and single yaml definitions."""

        models_factory(num_multi_models, is_multi=True)
        for counter in range(1, num_single_models + 1):
            single_model_factory(f"model-{counter}")

        model_controller = ModelController(options, None)
        model_controller.scan_and_load_models_metadata()
        assert len(model_controller.models_info) == (num_multi_models + num_single_models)

    @pytest.mark.parametrize("is_multi", [True, False], ids=["multi", "single"])
    @pytest.mark.usefixtures(
        "mock_prerequisites", "mock_fetch_models_from_datarobot", "mock_model_version_exists"
    )
    def test_lookup_affected_models_by_a_push_to_the_main_branch(
        self,
        options,
        git_repo,
        init_repo_with_models_factory,
        common_filepath,
        is_multi,
    ):
        """Test lookup affected models with a push event to the Git repository main branch."""

        num_models = 3
        init_repo_with_models_factory(num_models, is_multi=is_multi)
        model_controller = ModelController(options, GitTool(GitHubEnv.workspace_path()))
        model_controller.scan_and_load_models_metadata()
        model_controller.collect_datarobot_model_files()

        # Change 1 - one common module
        make_a_change_and_commit(git_repo, [str(common_filepath)], 1)

        models_info = list(model_controller.models_info.values())
        for model_index in range(num_models):
            model_main_program_filepath = models_info[model_index].main_program_filepath()
            make_a_change_and_commit(
                git_repo, [model_main_program_filepath.resolved], 2 + model_index
            )

        head_git_sha = "HEAD"
        for last_provision_git_sha in [None, f"HEAD~{num_models}"]:
            with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}), patch.dict(
                os.environ, {"GITHUB_SHA": head_git_sha}
            ), patch.object(
                ModelController,
                "_get_latest_model_version_git_commit_ancestor",
                return_value=last_provision_git_sha,
            ):
                model_controller.lookup_affected_models_by_the_current_action()

            models_info = list(model_controller.models_info.values())
            assert (
                len(
                    [
                        m_info
                        for m_info in models_info
                        if m_info.is_affected_by_commit(datarobot_latest_model_version={"id": "12"})
                    ]
                )
                == num_models
            )

        for reference in range(1, num_models):
            head_git_sha = "HEAD"
            last_provision_git_sha = f"HEAD~{reference}"
            with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}), patch.dict(
                os.environ, {"GITHUB_SHA": head_git_sha}
            ), patch.object(
                ModelController,
                "_get_latest_model_version_git_commit_ancestor",
                return_value=last_provision_git_sha,
            ):
                model_controller.lookup_affected_models_by_the_current_action()

            num_affected_models = len(
                [
                    m_info
                    for m_info in models_info
                    if m_info.is_affected_by_commit(datarobot_latest_model_version={"id": "12"})
                ]
            )
            assert num_affected_models == reference

    @pytest.mark.usefixtures("no_models")
    def test_save_statistics(self, options, github_output):
        """A case to test custom inference model's statistics saving."""

        with patch("dr_client.DrClient"):
            model_controller = ModelController(options, repo=None)
            validate_metrics(github_output, constants.Label.MODELS, model_controller)


class TestCustomInferenceModelDeletion:
    """Contains unit-tests for model deletion."""

    @pytest.fixture
    def user_provided_id(self):
        """A fixture to return a fake user model ID."""

        return "user-provided-1111"

    @pytest.fixture
    def model_id(self):
        """A fixture to return a fake DataRobot model ID."""

        return "model-id-2222"

    @contextlib.contextmanager
    def _mock_local_models(self, user_provided_id):
        with patch.object(
            ModelController, "models_info", new_callable=PropertyMock
        ) as models_info_property, patch.object(
            ModelInfo, "user_provided_id", new_callable=PropertyMock
        ) as model_info_user_provided_id:
            models_info_property.return_value = {
                user_provided_id: ModelInfo("yaml-path", "model-path", None)
            }
            model_info_user_provided_id.return_value = user_provided_id
            yield

    @contextlib.contextmanager
    def _mock_fetched_models_that_do_not_exist_locally(self, user_provided_id, model_id):
        with patch.object(
            ModelController, "datarobot_models", new_callable=PropertyMock
        ) as datarobot_models:
            # Ensure it does not match to the local model definition
            non_existing_user_provided_id = f"from-dr-{user_provided_id}"
            datarobot_models.return_value = {
                non_existing_user_provided_id: DataRobotModel(
                    model={"id": model_id or str(ObjectId())}, latest_version=None
                )
            }
            yield

    @contextlib.contextmanager
    def _mock_fetched_deployments(self, model_id, has_deployment=False):
        with patch.object(DrClient, "fetch_custom_model_deployments") as fetched_deployments:
            fetched_deployments.return_value = (
                [{"id": "dddd", "customModel": {"id": model_id}}] if has_deployment else []
            )
            yield

    @pytest.mark.parametrize("event_name", ["pull_request", "push"])
    def test_models_deletion_without_allowed_input_arg(
        self, options, user_provided_id, model_id, event_name
    ):
        """Test a failure to delete a model when input argument does not allow it."""

        options.allow_model_deletion = False
        model_controller = ModelController(options, None)
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": event_name}), self._mock_local_models(
            user_provided_id
        ), self._mock_fetched_models_that_do_not_exist_locally(
            user_provided_id, model_id
        ), self._mock_fetched_deployments(
            model_id, has_deployment=False
        ):
            with pytest.raises(IllegalModelDeletion) as ex:
                model_controller.handle_deleted_models()
            exception_msg = str(ex)
            assert "Model deletion was configured as not being allowed" in exception_msg

    def test_models_deletion_for_pull_request_event_without_deployment(
        self, options, user_provided_id, model_id
    ):
        """Test an undeployed model deletion during a pull request event."""

        options.allow_model_deletion = True
        model_controller = ModelController(options, None)
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "pull_request"}), self._mock_local_models(
            user_provided_id
        ), self._mock_fetched_models_that_do_not_exist_locally(
            user_provided_id, model_id
        ), self._mock_fetched_deployments(
            model_id, has_deployment=False
        ):
            model_controller.handle_deleted_models()

    def test_models_deletion_for_pull_request_event_with_deployment(
        self, options, user_provided_id, model_id
    ):
        """Test a failure to delete a deployed model during a pull request event."""

        options.allow_model_deletion = True
        model_controller = ModelController(options, None)
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "pull_request"}), self._mock_local_models(
            user_provided_id
        ), self._mock_fetched_models_that_do_not_exist_locally(
            user_provided_id, model_id
        ), self._mock_fetched_deployments(
            model_id, has_deployment=True
        ):
            with pytest.raises(IllegalModelDeletion):
                model_controller.handle_deleted_models()

    def test_models_deletion_for_push_event_and_no_deployments(
        self, options, user_provided_id, model_id
    ):
        """
        Test a successful deletion of a model without existing deployments during push event.
        """

        options.allow_model_deletion = True
        model_controller = ModelController(options, None)
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}), patch.object(
            DrClient, "delete_custom_model_by_model_id"
        ) as mock_dr_client_delete_cm, self._mock_local_models(
            user_provided_id
        ), self._mock_fetched_models_that_do_not_exist_locally(
            user_provided_id, model_id
        ), self._mock_fetched_deployments(
            model_id, has_deployment=False
        ):
            model_controller.handle_deleted_models()
            mock_dr_client_delete_cm.assert_called_once()

    def test_models_deletion_for_push_event_and_deployment(
        self, options, user_provided_id, model_id
    ):
        """
        Test a successful deletion of an undeployed model with other existing deployments during
        a push event.
        """

        options.allow_model_deletion = True
        model_controller = ModelController(options, None)
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}), self._mock_local_models(
            user_provided_id
        ), self._mock_fetched_models_that_do_not_exist_locally(
            user_provided_id, model_id
        ), self._mock_fetched_deployments(
            model_id, has_deployment=True
        ):
            with pytest.raises(IllegalModelDeletion):
                model_controller.handle_deleted_models()


@pytest.mark.usefixtures(
    "mock_prerequisites", "mock_fetch_models_from_datarobot", "mock_handle_deleted_models"
)
class TestGlobPatterns:
    """Contains unit-test for glob patters."""

    @pytest.mark.usefixtures("common_path_with_code", "github_output")
    @pytest.mark.parametrize("num_models", [1, 2, 3])
    @pytest.mark.parametrize("is_multi", [True, False], ids=["multi", "single"])
    @pytest.mark.parametrize(
        "with_include_glob",
        [True, False],
        ids=["with-include-glob", "without-include-glob"],
    )
    @pytest.mark.parametrize(
        "with_exclude_glob",
        [True, False],
        ids=["with-exclude-glob", "without-exclude-glob"],
    )
    def test_glob_patterns(
        self,
        models_factory,
        excluded_src_path,
        options,
        num_models,
        is_multi,
        common_filepath,
        with_include_glob,
        with_exclude_glob,
    ):
        """Test include Glob patterns and defaults in a given model definition."""

        models_factory(num_models, is_multi, with_include_glob, with_exclude_glob)
        custom_inference_model_action = CustomModelsAction(options)

        with patch.object(ModelController, "handle_model_changes"), patch.object(
            DeploymentController, "fetch_deployments_from_datarobot"
        ), patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}):
            custom_inference_model_action.run()

        assert len(custom_inference_model_action.model_controller.models_info) == num_models

        for _, model_info in custom_inference_model_action.model_controller.models_info.items():
            model_path = model_info.model_path
            readme_file_path = model_path / "README.md"
            assert readme_file_path.is_file()

            model_file_paths = [path.filepath for path in model_info.model_file_paths.values()]
            assert excluded_src_path not in model_file_paths

            if with_include_glob:
                assert common_filepath.absolute() in model_file_paths
            else:
                assert common_filepath.absolute() not in model_file_paths

            if with_exclude_glob:
                assert readme_file_path.absolute() not in model_file_paths
            else:
                assert readme_file_path.absolute() in model_file_paths

    @pytest.mark.parametrize(
        "model_path, included_paths, excluded_paths, expected_num_model_files",
        [
            ("/m", {"/m/custom.py", "/m", "/m/score/bbb.md"}, {}, 2),
            ("/m", {"/m/custom.py", "/m/.", "/m/score/bbb.md"}, {}, 2),
            ("/m", {"/m/custom.py", "/m/bbb.py", "/m/score/bbb.md"}, {}, 3),
            ("/m", {"/m/./custom.py", "/m/./bbb.py", "/m/./score/bbb.md"}, {"/m/bbb.py"}, 2),
            ("/m", {"/m/./custom.py", "/m/.//bbb.py", "/m/score/bbb.md"}, {"/m/bbb.py"}, 2),
            ("/m", {"/m/./custom.py", "/m/bbb.py", "/m/score/bbb.py"}, {"/m/bbb.py"}, 2),
            ("/m", {"/m/./custom.py", "/m/bbb.py", "/m/score/bbb.py"}, {"bbb.sh"}, 3),
            ("/m", {"/m/./custom.py", "/m/bbb.py", "/m/score/./bbb.py"}, {"/m/score/bbb.py"}, 2),
            ("/m", {"/m/./custom.py", "/m/bbb.py", "/m/score//bbb.py"}, {"/m/score/bbb.py"}, 2),
            ("/m", {"/m/./custom.py", "/m/.//bbb.py", "/m/score/./bbb.py"}, {"/m/score/bbb.py"}, 2),
            ("/m", {"/m/./custom.py", "/m/score/../bbb.py"}, {"/m/score/bbb.py"}, 2),
            ("/m", {"/m/./custom.py", "/m/score/../bbb.py"}, {"/m/bbb.py"}, 1),
            ("/m", {"/m/./custom.py", "/m//score/bbb.py"}, {"/m//score/bbb.py"}, 1),
            ("/m", {"/m/./custom.py", "/m//score/./bbb.py"}, {"/m/score/bbb.py"}, 1),
            (
                "/deployments/../m",
                {
                    "/deployments/../m/custom.py",
                    "/deployments/../m",
                    "/deployments/../m/score/bbb.py",
                },
                {"/deployments/../m/score/bbb.py"},
                1,
            ),
        ],
    )
    def test_filtered_model_paths(
        self, model_path, included_paths, excluded_paths, expected_num_model_files
    ):
        """Test excluded Glob patterns in a given model definition."""

        model_info = ModelInfo("yaml-file-path", model_path, None)
        with patch("common.git_tool.Repo.init"), patch("model_controller.DrClient"), patch(
            "model_controller.ModelInfo.user_provided_id", new_callable=PropertyMock("123")
        ):
            ModelController._set_filtered_model_paths(
                model_info, included_paths, excluded_paths, workspace_path="/"
            )
            assert len(model_info.model_file_paths) == expected_num_model_files
            for excluded_path in excluded_paths:
                assert Path(excluded_path) not in model_info.model_file_paths

    @pytest.mark.usefixtures("common_path_with_code")
    @pytest.mark.parametrize("is_multi", [True, False], ids=["multi", "single"])
    def test_missing_main_program(self, models_factory, options, is_multi):
        """Test missing main program in a given model."""

        models_factory(1, is_multi, include_main_prog=False)
        model_controller = ModelController(options, GitTool(GitHubEnv.workspace_path()))
        model_controller.scan_and_load_models_metadata()
        with pytest.raises(ModelMainEntryPointNotFound):
            model_controller.collect_datarobot_model_files()

    @pytest.mark.parametrize(
        "local_paths, shared_paths, collision_expected",
        [
            (["/repo/model/custom.py", "/repo/model/util.py"], ["/repo/util.py"], True),
            (
                ["/repo/model/custom.py", "/repo/model/common/util.py"],
                ["/repo/common/util.py"],
                True,
            ),
            (["/repo/model/common/convert.py"], ["/repo/common/util.py"], False),
            (["/repo/model/common/util/convert.py"], ["/repo/common/util.py"], False),
            (
                ["/repo/model/custom.py", "/repo/model/score.py"],
                ["/repo/common/util.py", "/repo/common/common.py"],
                False,
            ),
        ],
        ids=[
            "file-collision",
            "common-package-collision",
            "common-package-no-collision1",
            "common-package-no-collision2",
            "shared-package-no-collision",
        ],
    )
    def test_local_and_shared_collisions(self, local_paths, shared_paths, collision_expected):
        """Test collisions between local and shared file paths in a given model definition."""

        workspace_path = "/repo"
        options = argparse.Namespace(
            webserver="www.dummy.com",
            api_token="abc",
            skip_cert_verification=True,
            branch="master",
        )
        with patch.object(
            ModelController, "models_info", new_callable=PropertyMock
        ) as mock_models_info_property, patch.object(
            ModelInfo, "model_file_paths", new_callable=PropertyMock
        ) as mock_model_file_paths_property, patch.dict(
            os.environ, {"GITHUB_WORKSPACE": workspace_path}
        ):
            model_path = Path(f"{workspace_path}/model")
            model_info = ModelInfo("yaml-path", model_path, None)
            mock_models_info_property.return_value = model_info
            model_file_paths_property = {
                p: ModelFilePath(p, model_path, GitHubEnv.workspace_path()) for p in local_paths
            }
            model_file_paths_property.update(
                {p: ModelFilePath(p, model_path, GitHubEnv.workspace_path()) for p in shared_paths}
            )
            mock_model_file_paths_property.return_value = model_file_paths_property
            with patch("common.git_tool.Repo.init"), patch("model_controller.DrClient"):
                model_controller = ModelController(options, None)
                if collision_expected:
                    with pytest.raises(SharedAndLocalPathCollision):
                        model_controller._validate_collision_between_local_and_shared(model_info)
                else:
                    model_controller._validate_collision_between_local_and_shared(model_info)
