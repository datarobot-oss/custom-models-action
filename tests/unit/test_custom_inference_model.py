import os
from argparse import Namespace

from pathlib import Path

from mock import patch
import pytest

from mock.mock import PropertyMock

from custom_inference_model import CustomInferenceModel
from custom_inference_model import ModelInfo
from common.exceptions import ModelMainEntryPointNotFound, SharedAndLocalPathCollision
from common.git_tool import GitTool


class TestCustomInferenceModel:
    @pytest.fixture
    def no_models(self, common_path_with_code):
        yield common_path_with_code

    @pytest.mark.usefixtures("no_models")
    def test_scan_and_load_no_models(self, options):
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_info) == 0

    @pytest.mark.parametrize("num_models", [1, 2, 3])
    def test_scan_and_load_models_from_multi_separate_yaml_files(
        self, options, single_model_factory, num_models
    ):
        for counter in range(1, num_models + 1):
            single_model_factory(f"model-{counter}", write_metadata=True)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        assert len(custom_inference_model._models_info) == num_models

    @pytest.mark.parametrize("num_models", [0, 1, 3])
    def test_scan_and_load_models_from_one_multi_models_yaml_file(
        self, options, models_factory, num_models
    ):
        models_factory(num_models, is_multi=True)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
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
        custom_inference_model._scan_and_load_datarobot_models_metadata()
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
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        custom_inference_model._collect_datarobot_model_files()

        # Change 1 - one common module
        TestGitTool.make_a_change_and_commit(git_repo, [str(common_filepath)], 1)

        models_info = custom_inference_model.models_info
        for model_index in range(num_models):
            model_main_program_filepath = models_info[model_index].main_program_filepath()
            TestGitTool.make_a_change_and_commit(
                git_repo, [model_main_program_filepath], 2 + model_index
            )

        head_git_sha = "HEAD"
        for last_provision_git_sha in [None, f"HEAD~{num_models}"]:
            with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}), patch.dict(
                os.environ, {"GITHUB_SHA": head_git_sha}
            ), patch.object(
                CustomInferenceModel,
                "_get_latest_provisioned_model_git_version",
                return_value={
                    "mainBranchCommitSha": last_provision_git_sha,
                    "pullRequestCommitSha": None,
                },
            ):
                custom_inference_model._lookup_affected_models_by_the_current_action()

            models_info = custom_inference_model.models_info
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
                "_get_latest_provisioned_model_git_version",
                return_value={
                    "mainBranchCommitSha": last_provision_git_sha,
                    "pullRequestCommitSha": None,
                },
            ):
                custom_inference_model._lookup_affected_models_by_the_current_action()

            num_affected_models = len(
                [m_info for m_info in models_info if m_info.is_affected_by_commit]
            )
            assert num_affected_models == reference


class TestGitTool:
    def test_changed_files_between_commits(
        self, options, repo_root_path, git_repo, init_repo_with_models_factory, common_filepath
    ):
        init_repo_with_models_factory(2, is_multi=False)
        custom_inference_model = CustomInferenceModel(options)
        custom_inference_model._scan_and_load_datarobot_models_metadata()
        custom_inference_model._collect_datarobot_model_files()

        self.make_a_change_and_commit(git_repo, [str(common_filepath)], 1)

        models_info = custom_inference_model.models_info
        first_model_main_program_filepath = models_info[0].main_program_filepath()
        self.make_a_change_and_commit(
            git_repo, [str(common_filepath), first_model_main_program_filepath], 2
        )

        repo_tool = GitTool(repo_root_path)
        changed_files1, deleted_files = repo_tool.find_changed_files("HEAD")
        assert len(changed_files1) == 2, changed_files1
        assert not deleted_files
        changed_files2, _ = repo_tool.find_changed_files("HEAD", "HEAD~1")
        assert len(changed_files2) == 2, changed_files2
        assert set(changed_files2) == set(changed_files1)

        changed_files3, _ = repo_tool.find_changed_files("HEAD~1", "HEAD~2")
        assert len(changed_files3) == 1, changed_files2

        changed_files4, _ = repo_tool.find_changed_files("HEAD", "HEAD~2")
        assert len(changed_files4) == 2, changed_files4

    @staticmethod
    def make_a_change_and_commit(git_repo, file_paths, index):
        for file_path in file_paths:
            with open(file_path, "a") as f:
                f.write(f"\n# Automatic change ({index})")
        git_repo.index.add([str(f) for f in file_paths])
        git_repo.index.commit(f"Change number {index}")

    def test_is_ancestor_of(
        self, repo_root_path, git_repo, init_repo_with_models_factory, common_filepath
    ):
        init_repo_with_models_factory(1, is_multi=False)
        repo_tool = GitTool(repo_root_path)
        for index in range(1, 5):
            self.make_a_change_and_commit(git_repo, [str(common_filepath)], index)
            ancestor_ref = f"HEAD~{index}"
            assert repo_tool.is_ancestor_of(ancestor_ref, "HEAD")


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
    @pytest.mark.usefixtures("mock_prerequisites")
    @pytest.mark.usefixtures("mock_fetch_models_from_datarobot")
    def test_glob_patterns(
        self,
        models_factory,
        common_path_with_code,
        excluded_src_path,
        options,
        num_models,
        is_multi,
        with_include_glob,
        with_exclude_glob,
    ):
        models_factory(num_models, is_multi, with_include_glob, with_exclude_glob)
        custom_inference_model = CustomInferenceModel(options)

        with patch.object(CustomInferenceModel, "_lookup_affected_models_by_the_current_action"):
            custom_inference_model.run()

        assert len(custom_inference_model.models_info) == num_models

        for index in range(num_models):
            model_path = custom_inference_model.models_info[index].model_path
            readme_file_path = model_path / "README.md"
            assert readme_file_path.is_file()

            model_file_paths = custom_inference_model.models_info[index].model_file_paths
            assert excluded_src_path not in model_file_paths

            if with_include_glob:
                assert common_path_with_code in model_file_paths
            else:
                assert common_path_with_code not in model_file_paths

            if with_exclude_glob:
                assert readme_file_path.absolute() not in model_file_paths
            else:
                assert readme_file_path.absolute() in model_file_paths

    @pytest.mark.parametrize(
        "included_paths, excluded_paths, expected_num_model_files",
        [
            ({"./custom.py", "./bbb.py", "score/bbb.md"}, {"bbb.py"}, 2),
            ({"./custom.py", ".//bbb.py", "score/bbb.md"}, {"bbb.py"}, 2),
            ({"./custom.py", "bbb.py", "score/bbb.py"}, {"bbb.py"}, 2),
            ({"./custom.py", "bbb.py", "score/bbb.py"}, {"bbb.sh"}, 3),
            ({"./custom.py", "bbb.py", "score/./bbb.py"}, {"score/bbb.py"}, 2),
            ({"./custom.py", "bbb.py", "score//bbb.py"}, {"score/bbb.py"}, 2),
            ({"./custom.py", ".//bbb.py", "score/./bbb.py"}, {"score/bbb.py"}, 2),
            ({"./custom.py", "score/../bbb.py"}, {"score/bbb.py"}, 2),
            ({"./custom.py", "score/../bbb.py"}, {"bbb.py"}, 2),
            ({"./custom.py", "//score/bbb.py"}, {"/score/bbb.py"}, 1),
            ({"./custom.py", "//score/./bbb.py"}, {"/score/bbb.py"}, 1),
        ],
    )
    def test_filtered_model_paths(self, included_paths, excluded_paths, expected_num_model_files):
        model_info = ModelInfo("yaml-path", "model-path", None)
        CustomInferenceModel._set_filtered_model_paths(model_info, included_paths, excluded_paths)
        assert len(model_info.model_file_paths) == expected_num_model_files
        for excluded_path in excluded_paths:
            assert Path(excluded_path) not in model_info.model_file_paths

    @pytest.mark.parametrize("is_multi", [True, False], ids=["multi", "single"])
    @pytest.mark.usefixtures("mock_prerequisites")
    @pytest.mark.usefixtures("mock_fetch_models_from_datarobot")
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
            (["/repo/model/common/convert.py"], ["/repo/common/util.py"], True),
            (["/repo/model/common/util/convert.py"], ["/repo/common/util.py"], True),
            (
                ["/repo/model/custom.py", "/repo/model/score.py"],
                ["/repo/common/util.py", "/repo/common/common.py"],
                False,
            ),
        ],
        ids=[
            "file-collision",
            "package-collision1",
            "package-collision2",
            "package-collision3",
            "no-collision",
        ],
    )
    def test_local_and_shared_collisions(self, local_paths, shared_paths, collision_expected):
        options = Namespace(webserver="www.dummy.com", api_token="abc", root_dir="/repo")
        with patch.object(
            CustomInferenceModel, "models_info", new_callable=PropertyMock
        ) as mock_models_info_property, patch.object(
            ModelInfo, "model_file_paths", new_callable=PropertyMock
        ) as mock_model_file_paths_property:
            model_path = Path("/repo/model")
            model_info = ModelInfo("yaml-path", model_path, None)
            mock_models_info_property.return_value = model_info
            mock_model_file_paths_property.return_value = [Path(p) for p in local_paths] + [
                Path(p) for p in shared_paths
            ]
            with patch("common.git_tool.Repo.init"), patch("custom_inference_model.DrClient"):
                custom_inference_model = CustomInferenceModel(options)
                if collision_expected:
                    with pytest.raises(SharedAndLocalPathCollision):
                        custom_inference_model._validate_collision_between_local_and_shared(
                            model_info
                        )
                else:
                    custom_inference_model._validate_collision_between_local_and_shared(model_info)
