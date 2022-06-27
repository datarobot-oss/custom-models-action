import pytest

from common.convertors import MemoryConvertor
from common.exceptions import InvalidMemoryValue
from common.git_tool import GitTool
from tests.unit.conftest import make_a_change_and_commit


class TestConvertor:
    def test_to_bytes_success(self):
        for unit in MemoryConvertor.UNIT_TO_BYTES.keys():
            configured_memory = f"3{unit}"
            num_bytes = MemoryConvertor.to_bytes(configured_memory)
            assert 3 * MemoryConvertor.UNIT_TO_BYTES[unit] == num_bytes

    @pytest.mark.parametrize("invalid_configured_memory", ["3a", "3aM", "b3", "b3M", "1.2M", "3.3"])
    def test_to_bytes_failure(self, invalid_configured_memory):
        with pytest.raises(InvalidMemoryValue) as ex:
            MemoryConvertor.to_bytes(invalid_configured_memory)
        assert "The memory value format is invalid" in str(ex)


class TestGitTool:
    def test_changed_files_between_commits(
        self, options, git_repo, repo_root_path, init_repo_with_models_factory, common_filepath
    ):
        init_repo_with_models_factory(2, is_multi=False)

        make_a_change_and_commit(git_repo, [str(common_filepath)], 1)

        first_model_main_program_filepath = repo_root_path / "model_0" / "custom.py"
        assert first_model_main_program_filepath.is_file()
        make_a_change_and_commit(
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

    def test_is_ancestor_of(
        self, repo_root_path, git_repo, init_repo_with_models_factory, common_filepath
    ):
        init_repo_with_models_factory(1, is_multi=False)
        repo_tool = GitTool(repo_root_path)
        for index in range(1, 5):
            make_a_change_and_commit(git_repo, [str(common_filepath)], index)
            ancestor_ref = f"HEAD~{index}"
            assert repo_tool.is_ancestor_of(ancestor_ref, "HEAD")

    def test_merge_base_commit_sha(
        self, repo_root_path, git_repo, init_repo_with_models_factory, common_filepath
    ):
        init_repo_with_models_factory(1, is_multi=False)

        # 1. Create feature branch and checkout
        feature_branch = git_repo.create_head("feature")
        feature_branch.checkout()

        # 2. Make some changes in the feature branch
        for index in range(1, 4):
            make_a_change_and_commit(git_repo, [str(common_filepath)], index)

        head_sha = git_repo.head.commit.hexsha

        repo_tool = GitTool(repo_root_path)
        split_commit_sha = repo_tool.merge_base_commit_sha("master", head_sha)
        assert split_commit_sha == git_repo.heads["master"].commit.hexsha
