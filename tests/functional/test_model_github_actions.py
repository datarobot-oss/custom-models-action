import contextlib
from enum import Enum
import os
from pathlib import Path

import pytest

from common.convertors import MemoryConvertor
from dr_client import DrClient
from schema_validator import ModelSchema
from tests.functional.conftest import cleanup_models
from tests.functional.conftest import increase_model_memory_by_1mb
from tests.functional.conftest import run_github_action
from tests.functional.conftest import temporarily_replace_schema_value
from tests.functional.conftest import printout
from tests.functional.conftest import unique_str
from tests.functional.conftest import upload_and_update_dataset
from tests.functional.conftest import webserver_accessible


@pytest.fixture
def cleanup(dr_client, repo_root_path):
    yield

    cleanup_models(dr_client, repo_root_path)


@pytest.mark.skipif(not webserver_accessible(), reason="DataRobot webserver is not accessible.")
@pytest.mark.usefixtures("build_repo_for_testing", "set_model_dataset_for_testing")
class TestModelGitHubActions:
    class Change(Enum):
        INCREASE_MEMORY = 1
        ADD_FILE = 2
        REMOVE_FILE = 3
        DELETE_MODEL = 4

    @contextlib.contextmanager
    def enable_custom_model_testing(self, model_metadata_yaml_file, model_metadata):
        with temporarily_replace_schema_value(
            model_metadata_yaml_file,
            ModelSchema.TEST_KEY,
            ModelSchema.TEST_SKIP_KEY,
            new_value=False,
        ):
            yield

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
        printout("Create a feature branch ...")
        feature_branch = git_repo.create_head(feature_branch_name)

        # 2. Make changes, one at a time on a feature branch
        printout(
            "Make several changes on a feature branch, one at a time ... "
            f"{[c.name for c in changes]}"
        )
        for change in changes:
            # 3. Checkout feature branch
            feature_branch.checkout()

            # 4. Make a change and commit it
            if change == self.Change.INCREASE_MEMORY:
                printout("Increase the model memory ...")
                new_memory = increase_model_memory_by_1mb(model_metadata_yaml_file)
                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", f"Increase memory to {new_memory}")
            elif change == self.Change.ADD_FILE:
                printout("Add a new file to the mode ...")
                for filepath in files_to_add_and_remove:
                    with open(filepath, "w") as f:
                        f.write("# New file for testing")
                    git_repo.git.add(filepath)
                git_repo.git.commit("-m", "Add new files.")
            elif change == self.Change.REMOVE_FILE:
                printout("Remove files from the model ...")
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
            printout("Run custom model GitHub action (pull-request)...")
            run_github_action(
                repo_root_path,
                git_repo,
                main_branch_name,
                "pull_request",
                main_branch_head_sha=merge_branch_name,
                is_deploy=False,
            )

            # 8. Validation
            printout("Validate the change ...")
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
        printout("Merge the feature branch ...")
        git_repo.git.merge(merge_branch, "--squash")
        git_repo.git.add("--all")
        git_repo.git.commit("-m", "Changes from merged feature branch")

        printout("Run custom model GitHub action (push event) with testing ...")
        with self.enable_custom_model_testing(model_metadata_yaml_file, model_metadata):
            run_github_action(repo_root_path, git_repo, main_branch_name, "push", is_deploy=False)

        # 12. Validation
        printout("Validate ...")
        cm_version = dr_client.fetch_custom_model_latest_version_by_git_model_id(
            model_metadata[ModelSchema.MODEL_ID_KEY]
        )
        # Assuming 'INCREASE_MEMORY` change took place
        assert cm_version["maximumMemory"] == MemoryConvertor.to_bytes(new_memory)
        printout("Done")

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
        printout(
            f"Make several changes in a pull request, one at a time ... {[c.name for c in changes]}"
        )
        for change in changes:
            # 3. Checkout feature branch
            feature_branch.checkout()

            # 4. Make a change and commit it
            if change == self.Change.INCREASE_MEMORY:
                printout("Increase the model memory ...")
                new_memory = increase_model_memory_by_1mb(model_metadata_yaml_file)
                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", f"Increase memory to {new_memory}")
            elif change == self.Change.DELETE_MODEL:
                printout("Delete the model ...")
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
            printout("Run custom model GitHub action (pull-request) ...")
            run_github_action(
                repo_root_path,
                git_repo,
                main_branch_name,
                "pull_request",
                main_branch_head_sha=merge_branch_name,
                is_deploy=False,
            )

            # 8. Validation
            printout("Validate the change ...")
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
        printout("Merge to the main branch ...")
        git_repo.git.merge(merge_branch, "--squash")
        git_repo.git.add("--all")
        git_repo.git.commit("-m", "Changes from merged feature branch")
        printout("Run custom model GitHub action (push event) ...")
        run_github_action(repo_root_path, git_repo, main_branch_name, "push", is_deploy=False)

        # 12. Validation. The model is actually deleted only upon merging.
        printout("Validate after merging ...")
        assert change == self.Change.DELETE_MODEL
        models = dr_client.fetch_custom_models()
        if models:
            assert all(
                m.get("gitModelId") != model_metadata[ModelSchema.MODEL_ID_KEY] for m in models
            )
        printout("Done")

    @pytest.mark.usefixtures("cleanup")
    def test_e2e_push_event_with_multiple_changes(
        self, repo_root_path, git_repo, model_metadata_yaml_file, main_branch_name
    ):
        # 1. Make three changes, one at a time on the main branch
        printout("Make 3 changes one at a time on the main branch ...")
        for index in range(3):
            # 2. Make a change and commit it
            printout(f"Increase memory ... {index + 1}")
            new_memory = increase_model_memory_by_1mb(model_metadata_yaml_file)
            git_repo.git.add(model_metadata_yaml_file)
            git_repo.git.commit("-m", f"Increase memory to {new_memory}")

            # 3. Run GitHub pull request action
            printout("Run custom model GitHub action (push event) ...")
            run_github_action(repo_root_path, git_repo, main_branch_name, "push", is_deploy=False)
        printout("Done")

    def test_is_accessible(self):
        assert webserver_accessible()

    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_set_training_and_holdout_datasets_for_structured_model(
        self,
        dr_client,
        repo_root_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
    ):
        # 1. Create a model just as a preliminary requirement (use GitHub action)
        printout(
            "Create a custom model as a preliminary requirement. "
            "Run custom model GitHub action (push event) ..."
        )
        run_github_action(repo_root_path, git_repo, main_branch_name, "push", is_deploy=False)

        printout("Upload training with holdout dataset for structured model.")
        datasets_root = Path(__file__).parent / ".." / "datasets"
        training_and_holdout_dataset_filepath = (
            datasets_root / "juniors_3_year_stats_regression_structured_training_with_holdout.csv"
        )
        with upload_and_update_dataset(
            dr_client,
            training_and_holdout_dataset_filepath,
            model_metadata_yaml_file,
            ModelSchema.TRAINING_DATASET_KEY,
        ) as training_dataset_id:
            partition_column = "partitioning"
            with temporarily_replace_schema_value(
                model_metadata_yaml_file,
                ModelSchema.SETTINGS_SECTION_KEY,
                ModelSchema.PARTITIONING_COLUMN_KEY,
                new_value=partition_column,
            ):
                try:
                    git_repo.git.add(model_metadata_yaml_file)
                    git_repo.git.commit("-m", f"Update training / holdout dataset(s)")

                    printout("Run custom inference models GitHub action ...")
                    run_github_action(
                        repo_root_path, git_repo, main_branch_name, "push", is_deploy=False
                    )

                    # Validate
                    git_model_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)
                    custom_model = dr_client.fetch_custom_model_by_git_id(git_model_id)
                    assert custom_model["trainingDatasetId"] == training_dataset_id
                    assert custom_model["trainingDataPartitionColumn"] == partition_column
                finally:
                    cleanup_models(dr_client, repo_root_path)

        printout("Done")

    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_set_training_and_holdout_datasets_for_unstructured_model(
        self,
        dr_client,
        repo_root_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
    ):
        with temporarily_replace_schema_value(
            model_metadata_yaml_file,
            ModelSchema.TARGET_TYPE_KEY,
            new_value=ModelSchema.TARGET_TYPE_UNSTRUCTURED_OTHER_KEY,
        ):
            # 1. Create a model just as a preliminary requirement (use GitHub action)
            printout(
                "Create a custom model as a preliminary requirement. "
                "Run custom model GitHub action (push event) ..."
            )
            run_github_action(repo_root_path, git_repo, main_branch_name, "push", is_deploy=False)

            git_model_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)

            printout("Upload training and holdout datasets for unstructured model.")
            datasets_root = Path(__file__).parent / ".." / "datasets"
            training_dataset_filepath = (
                datasets_root / "juniors_3_year_stats_regression_unstructured_training.csv"
            )
            holdout_dataset_filepath = (
                datasets_root / "juniors_3_year_stats_regression_unstructured_holdout.csv"
            )
            with upload_and_update_dataset(
                dr_client,
                training_dataset_filepath,
                model_metadata_yaml_file,
                ModelSchema.TRAINING_DATASET_KEY,
            ) as training_dataset_id, upload_and_update_dataset(
                dr_client,
                holdout_dataset_filepath,
                model_metadata_yaml_file,
                ModelSchema.HOLDOUT_DATASET_KEY,
            ) as holdout_dataset_id:
                try:
                    git_repo.git.add(model_metadata_yaml_file)
                    git_repo.git.commit("-m", f"Update training / holdout dataset(s)")

                    printout("Run custom inference models GitHub action ...")
                    run_github_action(
                        repo_root_path, git_repo, main_branch_name, "push", is_deploy=False
                    )

                    # Validation
                    custom_model = dr_client.fetch_custom_model_by_git_id(git_model_id)
                    assert (
                        custom_model["externalMlopsStatsConfig"]["trainingDatasetId"]
                        == training_dataset_id
                    )
                    assert (
                        custom_model["externalMlopsStatsConfig"]["holdoutDatasetId"]
                        == holdout_dataset_id
                    )
                finally:
                    cleanup_models(dr_client, repo_root_path)

        printout("Done")

    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_update_model_settings(
        self,
        dr_client,
        repo_root_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
    ):
        git_model_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)

        # 1. Create a model just as a preliminary requirement (use GitHub action)
        printout(
            "Create a custom model as a preliminary requirement. "
            "Run custom model GitHub action (push event) ..."
        )
        run_github_action(repo_root_path, git_repo, main_branch_name, "push", is_deploy=False)

        unique_string = unique_str()
        for settings_key, desired_settings_value in [
            (ModelSchema.NAME_KEY, f"Some another name {unique_string}"),
            (ModelSchema.DESCRIPTION_KEY, f"Some unique desc {unique_string}"),
            (ModelSchema.LANGUAGE_KEY, "Legacy"),
            (ModelSchema.TARGET_NAME_KEY, "XBH/AB_jr"),  # Taken from the associated dataset
            (ModelSchema.PREDICTION_THRESHOLD_KEY, 0.2),  # Assuming the model type is regression
        ]:
            custom_model = dr_client.fetch_custom_model_by_git_id(git_model_id)
            actual_settings_value = custom_model[DrClient.MODEL_SETTINGS_KEYS_MAP[settings_key]]
            assert (
                desired_settings_value != actual_settings_value,
                f"Desired settings value '{desired_settings_value}' should be differ than the "
                f"actual '{actual_settings_value}'.",
            )

            with temporarily_replace_schema_value(
                model_metadata_yaml_file,
                ModelSchema.SETTINGS_SECTION_KEY,
                settings_key,
                new_value=desired_settings_value,
            ):
                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", f"Update model settings '{settings_key}'")

                printout("Run custom inference models GitHub action (push) ...")
                run_github_action(
                    repo_root_path, git_repo, main_branch_name, "push", is_deploy=False
                )

                # Validate
                custom_model = dr_client.fetch_custom_model_by_git_id(git_model_id)
                actual_settings_value = custom_model[DrClient.MODEL_SETTINGS_KEYS_MAP[settings_key]]
                assert (
                    desired_settings_value == actual_settings_value,
                    f"Desired settings value '{desired_settings_value}' should be equal to the "
                    f"actual '{actual_settings_value}'.",
                )
