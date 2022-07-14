from enum import Enum
import os

import pytest
import yaml

from common.convertors import MemoryConvertor
from common.exceptions import DataRobotClientError
from common.exceptions import IllegalModelDeletion
from schema_validator import ModelSchema
from tests.functional.conftest import increase_model_memory_by_1mb
from tests.functional.conftest import run_github_action
from tests.functional.conftest import webserver_accessible


@pytest.fixture
def feature_branch_name():
    return "feature"


@pytest.fixture
def merge_branch_name():
    return "merge-feature-branch"


@pytest.fixture
def cleanup(dr_client, model_metadata):
    yield

    try:
        dr_client.delete_custom_model_by_git_model_id(model_metadata[ModelSchema.MODEL_ID_KEY])
    except (IllegalModelDeletion, DataRobotClientError):
        pass


@pytest.mark.skipif(not webserver_accessible(), reason="DataRobot webserver is not accessible.")
@pytest.mark.usefixtures("build_repo_for_testing", "upload_dataset_for_testing")
class TestModelGitHubActions:
    class Change(Enum):
        INCREASE_MEMORY = 1
        ADD_FILE = 2
        REMOVE_FILE = 3
        DELETE_MODEL = 4

    @pytest.mark.usefixtures("cleanup")
    def test_e2e_pull_request_event_with_multiple_changes(
        self,
        dr_client,
        repo_root_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
        feature_branch_name,
        merge_branch_name,
    ):
        files_to_add_and_remove = [
            model_metadata_yaml_file.parent / "some_new_file_1.py",
            model_metadata_yaml_file.parent / "some_new_file_2.py",
        ]
        changes = [self.Change.INCREASE_MEMORY, self.Change.ADD_FILE, self.Change.REMOVE_FILE]
        # Ensure that the `INCREASE_MEMORY` is always first
        assert changes[0] == self.Change.INCREASE_MEMORY
        # Ensure that the `REMOVE_FILE` is always last
        assert changes[-1] == self.Change.REMOVE_FILE

        # 1. Create feature branch
        feature_branch = git_repo.create_head(feature_branch_name)

        # 2. Make changes, one at a time on a feature branch
        for change in changes:
            # 3. Checkout feature branch
            feature_branch.checkout()

            # 4. Make a change and commit it
            if change == self.Change.INCREASE_MEMORY:
                new_memory = increase_model_memory_by_1mb(model_metadata_yaml_file)
                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", f"Increase memory to {new_memory}")
            elif change == self.Change.ADD_FILE:
                for filepath in files_to_add_and_remove:
                    with open(filepath, "w") as f:
                        f.write("# New file for testing")
                    git_repo.git.add(filepath)
                git_repo.git.commit("-m", "Add new files.")
            elif change == self.Change.REMOVE_FILE:
                for filepath in files_to_add_and_remove:
                    os.remove(filepath)
                    git_repo.git.add(filepath)
                git_repo.git.commit("-m", f"Remove the files.")

            # 5. Create merge branch from master and check it out
            merge_branch = git_repo.create_head(merge_branch_name, main_branch_name)
            git_repo.head.reference = merge_branch
            git_repo.head.reset(index=True, working_tree=True)

            # 6. Merge feature branch --no-ff
            git_repo.git.merge(feature_branch, "--no-ff")

            # 7. Run GitHub pull request action
            run_github_action(
                repo_root_path,
                git_repo,
                main_branch_name,
                merge_branch_name,
                "pull_request",
                is_deploy=False,
            )

            # 8. Validation
            cm_version = dr_client.fetch_custom_model_latest_version_by_git_model_id(
                model_metadata[ModelSchema.MODEL_ID_KEY]
            )
            # Assuming `INCREASE_MEMORY` always first
            assert cm_version["maximumMemory"] == MemoryConvertor.to_bytes(new_memory)
            if change == self.Change.ADD_FILE:
                for filepath in files_to_add_and_remove:
                    assert filepath.name in [item["filePath"] for item in cm_version["items"]]
            elif change == self.Change.REMOVE_FILE:
                for filepath in files_to_add_and_remove:
                    assert filepath.name not in [item["filePath"] for item in cm_version["items"]]

            # 9. Checkout the main branch
            git_repo.heads.master.checkout()
            if change != changes[-1]:
                # 10. Delete the merge branch
                git_repo.delete_head(merge_branch, "--force")

        # 11. Merge changes from the merge branch into the main branch
        git_repo.git.merge(merge_branch, "--squash")
        git_repo.git.add("--all")
        git_repo.git.commit("-m", "Changes from merged feature branch")
        head_commit_sha = git_repo.head.commit.hexsha
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=False
        )

        # 12. Validation
        cm_version = dr_client.fetch_custom_model_latest_version_by_git_model_id(
            model_metadata[ModelSchema.MODEL_ID_KEY]
        )
        # Assuming 'INCREASE_MEMORY` change took place
        assert cm_version["maximumMemory"] == MemoryConvertor.to_bytes(new_memory)

    @pytest.mark.usefixtures("cleanup")
    def test_e2e_pull_request_event_with_model_deletion(
        self,
        dr_client,
        repo_root_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
        feature_branch_name,
        merge_branch_name,
    ):
        """
        This test first creates a PR with a simple change in order to create the model in
        DataRobot. Afterwards, it creates a another PR to delete the model definition, which
        should delete the model in DataRobot.
        """
        changes = [self.Change.INCREASE_MEMORY, self.Change.DELETE_MODEL]

        # 1. Create a feature branch
        feature_branch = git_repo.create_head(feature_branch_name)

        # 2. Make changes, one at a time on a feature branch
        for change in changes:
            # 3. Checkout feature branch
            feature_branch.checkout()

            # 4. Make a change and commit it
            if change == self.Change.INCREASE_MEMORY:
                new_memory = increase_model_memory_by_1mb(model_metadata_yaml_file)
                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", f"Increase memory to {new_memory}")
            elif change == self.Change.DELETE_MODEL:
                os.remove(model_metadata_yaml_file)
                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", f"Delete the model definition file")

            # 5. Create merge branch from master and check it out
            merge_branch = git_repo.create_head(merge_branch_name, main_branch_name)
            git_repo.head.reference = merge_branch
            git_repo.head.reset(index=True, working_tree=True)

            # 6. Merge feature branch --no-ff
            git_repo.git.merge(feature_branch, "--no-ff")

            # 7. Run GitHub pull request action
            run_github_action(
                repo_root_path,
                git_repo,
                main_branch_name,
                merge_branch_name,
                "pull_request",
                is_deploy=False,
            )

            # 8. Validation
            if change == self.Change.INCREASE_MEMORY:
                cm_version = dr_client.fetch_custom_model_latest_version_by_git_model_id(
                    model_metadata[ModelSchema.MODEL_ID_KEY]
                )
                # Assuming `INCREASE_MEMORY` always first
                assert cm_version["maximumMemory"] == MemoryConvertor.to_bytes(new_memory)
            elif change == self.Change.DELETE_MODEL:
                # The model is not deleted in the pull request, but only after merging.
                pass
            else:
                assert False, f"Unexpected changed: '{change.name}'"

            # 9. Checkout the main branch
            git_repo.heads.master.checkout()
            if change != changes[-1]:
                # 10. Delete the merge branch only if there are yet more changes to apply
                git_repo.delete_head(merge_branch, "--force")

        # 11. Merge changes from the merge branch into the main branch
        git_repo.git.merge(merge_branch, "--squash")
        git_repo.git.add("--all")
        git_repo.git.commit("-m", "Changes from merged feature branch")
        head_commit_sha = git_repo.head.commit.hexsha
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=False
        )

        # 12. Validation. The model is actually deleted only upon merging.
        assert change == self.Change.DELETE_MODEL
        models = dr_client.fetch_custom_models()
        if models:
            assert all(
                m.get("gitModelId") != model_metadata[ModelSchema.MODEL_ID_KEY] for m in models
            )

    @pytest.mark.usefixtures("cleanup")
    def test_e2e_push_event_with_multiple_changes(
        self, repo_root_path, git_repo, model_metadata_yaml_file, main_branch_name
    ):
        # 1. Make three changes, one at a time on the main branch
        for _ in range(3):
            # 2. Make a change and commit it
            new_memory = increase_model_memory_by_1mb(model_metadata_yaml_file)
            git_repo.git.add(model_metadata_yaml_file)
            git_repo.git.commit("-m", f"Increase memory to {new_memory}")

            # 3. Run GitHub pull request action
            head_commit_sha = git_repo.head.commit.hexsha
            run_github_action(
                repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=False
            )

    def test_is_accessible(self):
        assert webserver_accessible()
