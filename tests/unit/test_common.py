#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=too-many-arguments

"""A module that contains unit-tests for the common package."""
import re

import pytest

from common.convertors import MemoryConvertor
from common.exceptions import InvalidMemoryValue
from common.exceptions import NamespaceAlreadySet
from common.exceptions import NamespaceNotInitialized
from common.git_tool import GitTool
from common.github_env import GitHubEnv
from common.namepsace import Namespace
from tests.unit.conftest import make_a_change_and_commit


class TestConvertor:
    """Contains the convertor unit-tests."""

    def test_to_bytes_success(self):
        """Test a successful bytes conversion."""

        for unit, unit_value in MemoryConvertor.UNIT_TO_BYTES.items():
            multiplier = 3
            configured_memory = f"{multiplier}{unit}"
            num_bytes = MemoryConvertor.to_bytes(configured_memory)
            assert multiplier * unit_value == num_bytes

    @pytest.mark.parametrize("invalid_configured_memory", ["3a", "3aM", "b3", "b3M", "1.2M", "3.3"])
    def test_to_bytes_failure(self, invalid_configured_memory):
        """Test a failure in bytes conversion."""

        with pytest.raises(InvalidMemoryValue) as ex:
            MemoryConvertor.to_bytes(invalid_configured_memory)
        assert "The memory value format is invalid" in str(ex)


class TestGitTool:
    """Contains Git tool unit-tests."""

    def test_changed_files_between_commits(
        self, git_repo, workspace_path, init_repo_with_models_factory, common_filepath
    ):
        """Test changed files between commits."""

        init_repo_with_models_factory(2, is_multi=False)

        make_a_change_and_commit(git_repo, [str(common_filepath)], 1)

        first_model_main_program_filepath = workspace_path / "model_0" / "custom.py"
        assert first_model_main_program_filepath.is_file()
        make_a_change_and_commit(
            git_repo, [str(common_filepath), first_model_main_program_filepath], 2
        )

        repo_tool = GitTool(workspace_path)
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

    def test_is_ancestor_of(
        self, workspace_path, git_repo, init_repo_with_models_factory, common_filepath
    ):
        """Test the check for commit ancestor."""

        init_repo_with_models_factory(1, is_multi=False)
        repo_tool = GitTool(workspace_path)
        for index in range(1, 5):
            make_a_change_and_commit(git_repo, [str(common_filepath)], index)
            ancestor_ref = f"HEAD~{index}"
            assert repo_tool.is_ancestor_of(ancestor_ref, "HEAD")

    def test_merge_base_commit_sha(
        self, workspace_path, git_repo, init_repo_with_models_factory, common_filepath
    ):
        """Test the merge base commit sha."""

        init_repo_with_models_factory(1, is_multi=False)

        # 1. Create feature branch and checkout
        feature_branch = git_repo.create_head("feature")
        feature_branch.checkout()

        # 2. Make some changes in the feature branch
        for index in range(1, 4):
            make_a_change_and_commit(git_repo, [str(common_filepath)], index)

        head_sha = git_repo.head.commit.hexsha

        repo_tool = GitTool(workspace_path)
        split_commit_sha = repo_tool.merge_base_commit_sha("master", head_sha)
        assert split_commit_sha == git_repo.heads["master"].commit.hexsha


class TestNamespace:
    """A class to test the namespace module."""

    @pytest.fixture(scope="function", autouse=True)
    def _namespace_cleanup_and_recover(self):
        """A fixture to clean up a namespace if it was set."""

        origin_namespace = Namespace.namespace()
        if not origin_namespace:
            return

        try:
            Namespace.uninit()
            yield
        finally:
            Namespace.uninit()
            Namespace.init(origin_namespace)

    def test_init_namespace_success(self):
        """Test the namespace init method."""

        namespace = "dev"
        Namespace.init(namespace)
        assert Namespace.namespace() == f"{namespace}/"

    def test_init_with_namespace_multiple_times_success(self):
        """Test multiple set up of the same namespace."""

        namespace = "dev"
        for _ in range(3):
            Namespace.init(namespace)
            assert Namespace.namespace() == f"{namespace}/"

    @pytest.mark.parametrize("namespace", [None, ""], ids=["none", "empty-str"])
    def test_init_empty_namespace_success(self, namespace):
        """Test a successful init with empty namespace, which results in a default namespace."""

        Namespace.init(namespace)
        assert Namespace.namespace() == Namespace.default_namespace()

    def test_already_init_failure(self):
        """Test a failure in an attempt to initialize with different namespaces."""

        namespace = "dev-1"
        Namespace.init(namespace)
        with pytest.raises(NamespaceAlreadySet):
            Namespace.init(f"{namespace}-2")

    def test_namespace_init_after_uninit_success(self):
        """Test a failure in an attempt to set a namespace second time."""

        namespace = "dev-1"
        Namespace.init(namespace)
        origin_namespace = Namespace.namespace()

        namespace = f"{namespace}-2"
        Namespace.uninit()
        Namespace.init(namespace)
        another_namespace = Namespace.namespace()

        assert another_namespace != origin_namespace
        assert another_namespace == f"{namespace}/"

    def test_is_in_namespace_success(self):
        """Test a successful check for user provided ID in a namespace."""

        namespace = "dev"
        user_provided_id = "my-awesome-id"
        Namespace.init(namespace)
        namespaced_user_provided_id = Namespace.namespaced(user_provided_id)
        assert Namespace.is_in_namespace(namespaced_user_provided_id)
        assert namespaced_user_provided_id.startswith(f"{namespace}/")

    def test_is_in_namespace_no_init_failure(self):
        """Test a failure to check if in namespace when not initialized."""

        with pytest.raises(NamespaceNotInitialized):
            Namespace.is_in_namespace("my-awesome-id")

    def test_is_in_namespace_with_default_namespace_success(self):
        """Test user provided ID in a global namespace when no namespace is provided."""

        user_provided_id = "my-awesome-id"
        Namespace.init()
        namespaced_user_provided_id = Namespace.namespaced(user_provided_id)
        assert Namespace.is_in_namespace(namespaced_user_provided_id)
        assert namespaced_user_provided_id.startswith(Namespace.default_namespace())

    def test_namespaced_success(self):
        """Test that user provided ID is successfully tagged with a namespace."""

        Namespace.init("dev")
        namespace = Namespace.namespace()
        user_provided_id = "my-awesome-id"
        assert Namespace.namespaced(user_provided_id).startswith(namespace)

    def test_namespaced_when_no_init_failure(self):
        """Test a failure to use the `namespaced` method when init was not called."""

        user_provided_id = "my-awesome-id"
        with pytest.raises(NamespaceNotInitialized):
            Namespace.namespaced(user_provided_id)

    def test_un_namespaced_success(self):
        """Test that a namespace if removed from a user provided ID."""

        Namespace.init("dev")
        namespace = Namespace.namespace()
        user_provided_id = "my-awesome-id"
        namespaced_user_provided_id = Namespace.namespaced(user_provided_id)
        assert namespaced_user_provided_id.startswith(namespace)
        un_namespaced_user_provided_id = Namespace.un_namespaced(namespaced_user_provided_id)
        assert not Namespace.un_namespaced(namespaced_user_provided_id).startswith(namespace)
        assert un_namespaced_user_provided_id == user_provided_id


@pytest.mark.usefixtures("github_output")
class TestGitHubEnv:
    """A class that contains unit-tests for the GitHubEnv module."""

    @pytest.mark.parametrize("value", [1, -1, 0.5, None, [2, 3], (4, 5), {"a": 1}])
    def test_set_output_param_non_str_value(self, value):
        """A case to test non-strings output values."""

        param_name = "some-non-str-value-param"
        GitHubEnv.set_output_param(param_name, value)
        with open(GitHubEnv.github_output(), "r", encoding="utf-8") as fd:
            content = fd.read()
            assert f"{param_name}={value}" in content

    @pytest.mark.parametrize("value", ["Hello", "Hello World", "\nHello World\n\n"])
    def test_set_output_param_single_line_str_value(self, value):
        """A case to test single line string output values."""

        param_name = "some-single-line-str-param"
        GitHubEnv.set_output_param(param_name, value)
        with open(GitHubEnv.github_output(), "r", encoding="utf-8") as fd:
            content = fd.read()
            assert f"{param_name}={value.strip()}" in content

    @pytest.mark.parametrize("value", ["Hello\nWorld", "\n\nHello\nWorld\n"])
    def test_set_output_param_multilines_str_value(self, value):
        """A case to test multilines string output values."""

        param_name = "some-multilines-str-param"
        GitHubEnv.set_output_param(param_name, value)
        with open(GitHubEnv.github_output(), "r", encoding="utf-8") as fd:
            content = fd.read()

            expected_block = f"{param_name}<<.*\n{value.strip()}\n.*\n"
            assert re.search(expected_block, content)

    def test_set_output_param_two_multilines_str_values(self):
        """A case to test more than one multilines string output values."""

        param_name_prefix = "some-multilines-str-param"
        value_prefix = "Hello\nWorld"
        for index in range(2):
            GitHubEnv.set_output_param(f"{param_name_prefix}-{index}", f"{value_prefix}-{index}")

        with open(GitHubEnv.github_output(), "r", encoding="utf-8") as fd:
            content = fd.read()

            for index in range(2):
                expected_block = f"{param_name_prefix}-{index}<<.*\n{value_prefix}-{index}\n.*\n"
                assert re.search(expected_block, content)
