import contextlib
import os
from argparse import Namespace

from pathlib import Path

from mock import patch
import pytest

from mock.mock import PropertyMock

from common.data_types import DataRobotModel
from common.git_tool import GitTool
from custom_inference_model import CustomInferenceModel
from custom_inference_model import ModelFilePath
from custom_inference_model import ModelInfo
from common.exceptions import (
    IllegalModelDeletion,
    ModelMainEntryPointNotFound,
    ModelMetadataAlreadyExists,
)
from common.exceptions import SharedAndLocalPathCollision
from dr_client import DrClient
from tests.unit.conftest import make_a_change_and_commit


class TestCustomInferenceModel:
    @pytest.fixture
    def no_models(self, common_path_with_code):
        yield common_path_with_code

    @pytest.mark.usefixtures("no_models")
    def test_scan_and_load_no_models(self, options):
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_models_metadata()
        assert len(custom_inference_model._models_info) == 0

    @pytest.mark.parametrize("num_models", [1, 2, 3])
    def test_scan_and_load_models_from_multi_separate_yaml_files(
        self, options, single_model_factory, num_models
    ):
        for counter in range(1, num_models + 1):
            single_model_factory(f"model-{counter}", write_metadata=True)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_models_metadata()
        assert len(custom_inference_model._models_info) == num_models

    def test_scan_and_load_models_with_same_git_model_id_failure(
        self, options, single_model_factory
    ):
        git_model_id = "same-git-model-id-111"
        single_model_factory(f"model-1", write_metadata=True, git_model_id=git_model_id)
        single_model_factory(f"model-2", write_metadata=True, git_model_id=git_model_id)
        custom_inference_model = CustomInferenceModel(options)
        with pytest.raises(ModelMetadataAlreadyExists):
            custom_inference_model._scan_and_load_models_metadata()

    @pytest.mark.parametrize("num_models", [0, 1, 3])
    def test_scan_and_load_models_from_one_multi_models_yaml_file(
        self, options, models_factory, num_models
    ):
        models_factory(num_models, is_multi=True)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_models_metadata()
        assert len(custom_inference_model._models_info) == num_models

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
        models_factory(num_multi_models, is_multi=True)
        for counter in range(1, num_single_models + 1):
            single_model_factory(f"model-{counter}")

        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_models_metadata()
        assert len(custom_inference_model._models_info) == (num_multi_models + num_single_models)

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
        num_models = 3
        init_repo_with_models_factory(num_models, is_multi=is_multi)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_models_metadata()
        custom_inference_model._collect_datarobot_model_files()

        # Change 1 - one common module
        make_a_change_and_commit(git_repo, [str(common_filepath)], 1)

        models_info = list(custom_inference_model.models_info.values())
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
                CustomInferenceModel,
                "_get_latest_model_version_git_commit_ancestor",
                return_value=last_provision_git_sha,
            ):
                custom_inference_model._lookup_affected_models_by_the_current_action()

            models_info = list(custom_inference_model.models_info.values())
            assert (
                len([m_info for m_info in models_info if m_info.is_affected_by_commit])
                == num_models
            )

        for reference in range(1, num_models):
            head_git_sha = "HEAD"
            last_provision_git_sha = f"HEAD~{reference}"
            with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}), patch.dict(
                os.environ, {"GITHUB_SHA": head_git_sha}
            ), patch.object(
                CustomInferenceModel,
                "_get_latest_model_version_git_commit_ancestor",
                return_value=last_provision_git_sha,
            ):
                custom_inference_model._lookup_affected_models_by_the_current_action()

            num_affected_models = len(
                [m_info for m_info in models_info if m_info.is_affected_by_commit]
            )
            assert num_affected_models == reference


class TestCustomInferenceModelDeletion:
    def test_models_deletion_without_allowed_input_arg(self, options):
        options.allow_model_deletion = False
        custom_inference_model = CustomInferenceModel(options)
        with patch.object(DrClient, "fetch_custom_model_deployments") as mock_fetch_deployments:
            custom_inference_model._handle_deleted_models()
            mock_fetch_deployments.assert_not_called()

    @pytest.fixture
    def git_model_id(self):
        return "git-model-1111"

    @pytest.fixture
    def model_id(self):
        return "model-id-2222"

    @contextlib.contextmanager
    def _mock_local_models(self, git_model_id):
        with patch.object(
            CustomInferenceModel, "models_info", new_callable=PropertyMock
        ) as models_info_property, patch.object(
            ModelInfo, "git_model_id", new_callable=PropertyMock
        ) as model_info_git_model_id:
            models_info_property.return_value = {
                git_model_id: ModelInfo("yaml-path", "model-path", None)
            }
            model_info_git_model_id.return_value = git_model_id
            yield

    @contextlib.contextmanager
    def _mock_fetched_models_that_do_not_exist_locally(self, git_model_id, model_id):
        with patch.object(
            CustomInferenceModel, "datarobot_models", new_callable=PropertyMock
        ) as datarobot_models:
            # Ensure it does not match to the local model definition
            non_existing_git_model_id = f"a{git_model_id}"
            datarobot_models.return_value = {
                non_existing_git_model_id: DataRobotModel(
                    model={"id": model_id}, latest_version=None
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

    def test_models_deletion_for_pull_request_event_without_deployment(
        self, options, git_model_id, model_id
    ):
        options.allow_model_deletion = True
        custom_inference_model = CustomInferenceModel(options)
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "pull_request"}), self._mock_local_models(
            git_model_id
        ), self._mock_fetched_models_that_do_not_exist_locally(
            git_model_id, model_id
        ), self._mock_fetched_deployments(
            model_id, has_deployment=False
        ):
            custom_inference_model._handle_deleted_models()

    def test_models_deletion_for_pull_request_event_with_deployment(
        self, options, git_model_id, model_id
    ):
        options.allow_model_deletion = True
        custom_inference_model = CustomInferenceModel(options)
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "pull_request"}), self._mock_local_models(
            git_model_id
        ), self._mock_fetched_models_that_do_not_exist_locally(
            git_model_id, model_id
        ), self._mock_fetched_deployments(
            model_id, has_deployment=True
        ):
            with pytest.raises(IllegalModelDeletion):
                custom_inference_model._handle_deleted_models()

    def test_models_deletion_for_push_event_and_no_deployments(
        self, options, git_model_id, model_id
    ):
        options.allow_model_deletion = True
        custom_inference_model = CustomInferenceModel(options)
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}), patch.object(
            DrClient, "delete_custom_model_by_model_id"
        ) as mock_dr_client_delete_cm, self._mock_local_models(
            git_model_id
        ), self._mock_fetched_models_that_do_not_exist_locally(
            git_model_id, model_id
        ), self._mock_fetched_deployments(
            model_id, has_deployment=False
        ):
            custom_inference_model._handle_deleted_models()
            mock_dr_client_delete_cm.assert_called_once()

    def test_models_deletion_for_push_event_and_deployment(self, options, git_model_id, model_id):
        options.allow_model_deletion = True
        custom_inference_model = CustomInferenceModel(options)
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}), patch.object(
            DrClient, "delete_custom_model_by_model_id"
        ) as mock_dr_client_delete_cm, self._mock_local_models(
            git_model_id
        ), self._mock_fetched_models_that_do_not_exist_locally(
            git_model_id, model_id
        ), self._mock_fetched_deployments(
            model_id, has_deployment=True
        ):
            custom_inference_model._handle_deleted_models()
            mock_dr_client_delete_cm.assert_not_called()


@pytest.mark.usefixtures(
    "mock_prerequisites", "mock_fetch_models_from_datarobot", "mock_handle_deleted_models"
)
class TestGlobPatterns:
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
        common_path_with_code,
        excluded_src_path,
        options,
        num_models,
        is_multi,
        common_filepath,
        with_include_glob,
        with_exclude_glob,
    ):
        models_factory(num_models, is_multi, with_include_glob, with_exclude_glob)
        custom_inference_model = CustomInferenceModel(options)

        with patch.object(CustomInferenceModel, "_lookup_affected_models_by_the_current_action"):
            custom_inference_model.run()

        assert len(custom_inference_model.models_info) == num_models

        for _, model_info in custom_inference_model.models_info.items():
            model_path = model_info.model_path
            readme_file_path = model_path / "README.md"
            assert readme_file_path.is_file()

            model_file_paths = model_info.model_file_paths
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
        "included_paths, excluded_paths, expected_num_model_files",
        [
            ({"/m/custom.py", "/m", "/m/score/bbb.md"}, {}, 2),
            ({"/m/custom.py", "/m/.", "/m/score/bbb.md"}, {}, 2),
            ({"/m/custom.py", "/m/bbb.py", "/m/score/bbb.md"}, {}, 3),
            ({"/m/./custom.py", "/m/./bbb.py", "/m/./score/bbb.md"}, {"/m/bbb.py"}, 2),
            ({"/m/./custom.py", "/m/.//bbb.py", "/m/score/bbb.md"}, {"/m/bbb.py"}, 2),
            ({"/m/./custom.py", "/m/bbb.py", "/m/score/bbb.py"}, {"/m/bbb.py"}, 2),
            ({"/m/./custom.py", "/m/bbb.py", "/m/score/bbb.py"}, {"bbb.sh"}, 3),
            ({"/m/./custom.py", "/m/bbb.py", "/m/score/./bbb.py"}, {"/m/score/bbb.py"}, 2),
            ({"/m/./custom.py", "/m/bbb.py", "/m/score//bbb.py"}, {"/m/score/bbb.py"}, 2),
            ({"/m/./custom.py", "/m/.//bbb.py", "/m/score/./bbb.py"}, {"/m/score/bbb.py"}, 2),
            ({"/m/./custom.py", "/m/score/../bbb.py"}, {"/m/score/bbb.py"}, 2),
            ({"/m/./custom.py", "/m/score/../bbb.py"}, {"/m/bbb.py"}, 1),
            ({"/m/./custom.py", "/m//score/bbb.py"}, {"/m//score/bbb.py"}, 1),
            ({"/m/./custom.py", "/m//score/./bbb.py"}, {"/m/score/bbb.py"}, 1),
        ],
    )
    def test_filtered_model_paths(self, included_paths, excluded_paths, expected_num_model_files):
        model_info = ModelInfo("yaml-path", "/m", None)
        with patch("common.git_tool.Repo.init"), patch("custom_inference_model.DrClient"), patch(
            "custom_inference_model.ModelInfo.git_model_id", new_callable=PropertyMock("123")
        ):
            CustomInferenceModel._set_filtered_model_paths(
                model_info, included_paths, excluded_paths, repo_root_dir="/"
            )
            assert len(model_info.model_file_paths) == expected_num_model_files
            for excluded_path in excluded_paths:
                assert Path(excluded_path) not in model_info.model_file_paths

    @pytest.mark.parametrize("is_multi", [True, False], ids=["multi", "single"])
    def test_missing_main_program(self, models_factory, common_path_with_code, options, is_multi):
        models_factory(1, is_multi, include_main_prog=False)
        custom_inference_model = CustomInferenceModel(options)
        with pytest.raises(ModelMainEntryPointNotFound):
            custom_inference_model.run()

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
        options = Namespace(
            webserver="www.dummy.com",
            api_token="abc",
            skip_cert_verification=True,
            root_dir="/repo",
        )
        with patch.object(
            CustomInferenceModel, "models_info", new_callable=PropertyMock
        ) as mock_models_info_property, patch.object(
            ModelInfo, "model_file_paths", new_callable=PropertyMock
        ) as mock_model_file_paths_property:
            model_path = Path("/repo/model")
            model_info = ModelInfo("yaml-path", model_path, None)
            mock_models_info_property.return_value = model_info
            model_file_paths_property = {
                p: ModelFilePath(p, model_path, options.root_dir) for p in local_paths
            }
            model_file_paths_property.update(
                {p: ModelFilePath(p, model_path, options.root_dir) for p in shared_paths}
            )
            mock_model_file_paths_property.return_value = model_file_paths_property
            with patch("common.git_tool.Repo.init"), patch("custom_inference_model.DrClient"):
                custom_inference_model = CustomInferenceModel(options)
                if collision_expected:
                    with pytest.raises(SharedAndLocalPathCollision):
                        custom_inference_model._validate_collision_between_local_and_shared(
                            model_info
                        )
                else:
                    custom_inference_model._validate_collision_between_local_and_shared(model_info)
