#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=too-many-arguments

"""
Functional tests for the custom inference model GitHub action. Functional tests are executed
against a running DataRobot application. If DataRobot is not accessible, the functional tests
are skipped.
"""

import contextlib
import os
import shutil
from enum import Enum
from pathlib import Path

import pytest

from common.convertors import MemoryConvertor
from common.exceptions import DataRobotClientError
from dr_client import DrClient
from schema_validator import ModelSchema
from tests.conftest import unique_str
from tests.functional.conftest import cleanup_models
from tests.functional.conftest import increase_model_memory_by_1mb
from tests.functional.conftest import printout
from tests.functional.conftest import run_github_action
from tests.functional.conftest import temporarily_replace_schema_value
from tests.functional.conftest import (
    temporarily_upload_training_dataset_for_structured_model,
)
from tests.functional.conftest import upload_and_update_dataset
from tests.functional.conftest import webserver_accessible


@pytest.fixture
def cleanup(dr_client, workspace_path):
    """A fixture to delete models in DataRobot that were created from the local source tree."""

    yield

    cleanup_models(dr_client, workspace_path)


@pytest.mark.skipif(not webserver_accessible(), reason="DataRobot webserver is not accessible.")
@pytest.mark.usefixtures("build_repo_for_testing", "set_model_dataset_for_testing", "github_output")
class TestModelGitHubActions:
    """Contains an end-to-end test cases for the custom inference model GitHub action."""

    class Change(Enum):
        """An enum to indicate the type of change."""

        INCREASE_MEMORY = 1
        ADD_FILE = 2
        REMOVE_FILE = 3
        DELETE_MODEL = 4

    @staticmethod
    @contextlib.contextmanager
    def enable_custom_model_testing(model_metadata_yaml_file):
        """Temporarily enables custom model testing in DataRobot."""

        with temporarily_replace_schema_value(
            model_metadata_yaml_file,
            ModelSchema.TEST_KEY,
            ModelSchema.TEST_SKIP_KEY,
            new_value=False,
        ):
            yield

    @pytest.mark.usefixtures("cleanup")
    def test_e2e_pull_request_event_with_multiple_changes(  # pylint: disable=too-many-locals
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
        feature_branch_name,
        merge_branch_name,
    ):
        """
        And end-to-end case to test the custom inference model GitHub action when it is
        executed from a pull request branch with multiple commits.
        """

        printout("Create a feature branch ...")
        feature_branch = git_repo.create_head(feature_branch_name)
        checks = [self._increase_memory_check, self._add_file_check, self._remove_file_check]
        self._run_checks(
            checks,
            feature_branch,
            git_repo,
            workspace_path,
            main_branch_name,
            model_metadata,
            model_metadata_yaml_file,
            merge_branch_name,
            dr_client,
        )
        self._run_github_action_with_testing_enabled(
            dr_client,
            workspace_path,
            git_repo,
            main_branch_name,
            model_metadata,
            model_metadata_yaml_file,
        )
        printout("Done")

    @classmethod
    def _run_checks(
        cls,
        checks,
        feature_branch,
        git_repo,
        workspace_path,
        main_branch_name,
        model_metadata,
        model_metadata_yaml_file,
        merge_branch_name,
        dr_client,
    ):
        # Make changes, one at a time on a feature branch
        printout("Make several changes on a feature branch, one at a time.")
        merge_branch = None
        for check_method in checks:
            # Checkout the feature branch
            feature_branch.checkout()

            if merge_branch:
                # Delete the merge branch to enable creation of another merge branch
                git_repo.delete_head(merge_branch, "--force")

            with check_method(git_repo, dr_client, model_metadata, model_metadata_yaml_file):
                # Create merge branch from master and check it out
                merge_branch = git_repo.create_head(merge_branch_name, main_branch_name)
                git_repo.head.reference = merge_branch
                git_repo.head.reset(index=True, working_tree=True)

                # Merge feature branch --no-ff
                git_repo.git.merge(feature_branch, "--no-ff")

                # Run GitHub pull request action
                printout("Run custom model GitHub action (pull-request)...")
                run_github_action(
                    workspace_path,
                    git_repo,
                    main_branch_name,
                    event_name="pull_request",
                    is_deploy=False,
                    # main_branch_head_sha=merge_branch_name,
                )
        cls._merge_changes_into_the_main_branch(git_repo, merge_branch)

    @staticmethod
    @contextlib.contextmanager
    def _increase_memory_check(git_repo, dr_client, model_metadata, model_metadata_yaml_file):
        printout("Increase the model memory ...")
        new_memory = increase_model_memory_by_1mb(model_metadata_yaml_file)
        git_repo.git.add(model_metadata_yaml_file)
        git_repo.git.commit("-m", f"Increase memory to {new_memory}")

        yield

        printout("Validate the increase memory check")
        cm_version = dr_client.fetch_custom_model_latest_version_by_user_provided_id(
            model_metadata[ModelSchema.MODEL_ID_KEY]
        )

        assert cm_version["maximumMemory"] == MemoryConvertor.to_bytes(new_memory)

    @classmethod
    @contextlib.contextmanager
    def _add_file_check(cls, git_repo, dr_client, model_metadata, model_metadata_yaml_file):
        with cls._add_remove_file_check(
            dr_client, git_repo, model_metadata, model_metadata_yaml_file, is_add=True
        ):
            yield

    @classmethod
    @contextlib.contextmanager
    def _remove_file_check(cls, git_repo, dr_client, model_metadata, model_metadata_yaml_file):
        with cls._add_remove_file_check(
            dr_client, git_repo, model_metadata, model_metadata_yaml_file, is_add=False
        ):
            yield

    @staticmethod
    @contextlib.contextmanager
    def _add_remove_file_check(
        dr_client, git_repo, model_metadata, model_metadata_yaml_file, is_add=True
    ):
        printout("Add a new file to the mode ...")

        files_to_add_and_remove = [
            model_metadata_yaml_file.parent / "some_new_file_1.py",
            model_metadata_yaml_file.parent / "some_new_file_2.py",
        ]
        for filepath in files_to_add_and_remove:
            if is_add:
                with open(filepath, "w", encoding="utf-8") as fd:
                    fd.write("# New file for testing")
            else:
                os.remove(filepath)
            git_repo.git.add(filepath)
        commit_msg = "Add new files." if is_add else "Remove the files."
        git_repo.git.commit("-m", commit_msg)

        yield

        printout("Validate the increase memory check")
        cm_version = dr_client.fetch_custom_model_latest_version_by_user_provided_id(
            model_metadata[ModelSchema.MODEL_ID_KEY]
        )
        cm_version_files = [item["filePath"] for item in cm_version["items"]]
        for filepath in files_to_add_and_remove:
            assert (filepath.name in cm_version_files) == is_add

    @staticmethod
    def _merge_changes_into_the_main_branch(git_repo, merge_branch):
        # Merge changes from the merge branch into the main branch
        printout("Merge the feature branch ...")
        git_repo.heads.master.checkout()
        git_repo.git.merge(merge_branch, "--squash")
        git_repo.git.add("--all")
        git_repo.git.commit("-m", "Changes from merged feature branch")

    @classmethod
    def _run_github_action_with_testing_enabled(
        cls,
        dr_client,
        workspace_path,
        git_repo,
        main_branch_name,
        model_metadata,
        model_metadata_yaml_file,
    ):
        printout("Run custom model GitHub action (push event) with testing ...")
        with cls.enable_custom_model_testing(model_metadata_yaml_file):
            run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        custom_model = dr_client.fetch_custom_model_by_git_id(
            model_metadata[ModelSchema.MODEL_ID_KEY]
        )
        custom_model_tests = dr_client.fetch_custom_model_tests(custom_model["id"])
        assert len(custom_model_tests) == 1, custom_model_tests
        assert custom_model_tests[0]["overallStatus"] == "succeeded", custom_model_tests

    @pytest.mark.usefixtures("cleanup")
    def test_e2e_pull_request_event_with_model_deletion(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
        feature_branch_name,
        merge_branch_name,
    ):
        """
        An end-to-end case to test model deletion by the custom inference model GitHub action
        from a pull-request. The test first creates a PR with a simple change in order to create
        the model in DataRobot. Afterwards, it creates another PR to delete the model definition,
        which should delete the model in DataRobot.
        """

        # Create a feature branch
        printout("Create a feature branch ...")
        feature_branch = git_repo.create_head(feature_branch_name)

        # Make changes, one at a time on a feature branch
        printout("Make several changes on a feature branch, one at a time.")
        checks = [self._increase_memory_check, self._delete_model_check]
        self._run_checks(
            checks,
            feature_branch,
            git_repo,
            workspace_path,
            main_branch_name,
            model_metadata,
            model_metadata_yaml_file,
            merge_branch_name,
            dr_client,
        )

        printout("Run custom model GitHub action (push event) ...")
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        # 12. Validation. The model is actually deleted only upon merging.
        printout("Validate after merging ...")
        models = dr_client.fetch_custom_models()
        if models:
            assert all(
                model.get("userProvidedId") != model_metadata[ModelSchema.MODEL_ID_KEY]
                for model in models
            )
        printout("Done")

    @staticmethod
    @contextlib.contextmanager
    def _delete_model_check(git_repo, dr_client, model_metadata, model_metadata_yaml_file):
        # pylint: disable=unused-argument
        printout("Delete the model ...")
        os.remove(model_metadata_yaml_file)
        git_repo.git.add(model_metadata_yaml_file)
        git_repo.git.commit("-m", "Delete the model definition file")
        yield

    @pytest.mark.usefixtures("cleanup")
    def test_e2e_push_event_with_multiple_changes(
        self, workspace_path, git_repo, model_metadata_yaml_file, main_branch_name
    ):
        """
        An end-to-end case to test a push event with multiple commits, by the custom
        inference model GitHub action.
        """

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
            run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)
        printout("Done")

    def test_is_accessible(self):
        """A test case to check whether DataRobot webserver is accessible."""

        assert webserver_accessible()

    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_set_training_and_holdout_datasets_for_structured_model(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
    ):
        """
        And end-to-end case to test a training dataset assignment for structured model, by the
        custom inference model GitHub action. The training dataset contains a holdout column.
        """

        # 1. Create a model just as a preliminary requirement (use GitHub action)
        printout(
            "Create a custom model as a preliminary requirement. "
            "Run custom model GitHub action (push event) ..."
        )
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        with temporarily_upload_training_dataset_for_structured_model(
            dr_client, model_metadata_yaml_file, event_name="push"
        ) as (training_dataset_id, partition_column):
            try:
                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", "Update training / holdout dataset(s)")

                printout("Run custom inference models GitHub action ...")
                run_github_action(
                    workspace_path, git_repo, main_branch_name, "push", is_deploy=False
                )

                # Validate
                user_provided_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)
                custom_model = dr_client.fetch_custom_model_by_git_id(user_provided_id)
                assert custom_model["trainingDatasetId"] == training_dataset_id
                assert custom_model["trainingDataPartitionColumn"] == partition_column
            finally:
                cleanup_models(dr_client, workspace_path)

        printout("Done")

    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_set_training_dataset_with_wrong_model_target_name(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
    ):
        """
        And end-to-end case to test a training dataset assignment for structured model, whose
        definition initially contains wrong target name.
        """

        with temporarily_upload_training_dataset_for_structured_model(
            dr_client, model_metadata_yaml_file, event_name="push"
        ) as (training_dataset_id, partition_column):
            try:
                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", "Update training / holdout dataset(s)")

                printout(
                    "Create a custom model with wrong target name. "
                    "Run custom model GitHub action (push event) ..."
                )

                with temporarily_replace_schema_value(
                    model_metadata_yaml_file,
                    ModelSchema.SETTINGS_SECTION_KEY,
                    ModelSchema.TARGET_NAME_KEY,
                    new_value="wrong-target-name",
                ):
                    git_repo.git.add(model_metadata_yaml_file)
                    git_repo.git.commit("-m", "Set wrong target name")

                    with pytest.raises(DataRobotClientError) as ex:
                        run_github_action(
                            workspace_path, git_repo, main_branch_name, "push", is_deploy=False
                        )
                    assert ex.value.code == 422, ex.value
                    assert (
                        "Custom model's target is not found in the provided dataset"
                        in ex.value.args[0]
                    ), ex.value.args

                git_repo.git.add(model_metadata_yaml_file)
                git_repo.git.commit("-m", "Set valid target name")

                printout("Run custom inference models GitHub action with proper target ...")
                run_github_action(
                    workspace_path, git_repo, main_branch_name, "push", is_deploy=False
                )

                # Validate
                user_provided_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)
                custom_model = dr_client.fetch_custom_model_by_git_id(user_provided_id)
                assert custom_model["trainingDatasetId"] == training_dataset_id
                assert custom_model["trainingDataPartitionColumn"] == partition_column
            finally:
                cleanup_models(dr_client, workspace_path)

        printout("Done")

    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_set_training_and_holdout_datasets_for_unstructured_model(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
    ):
        """
        An end-to-end case to test training and holdout dataset assignment for unstructured
        model by the custom inference model GitHub action.
        """

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
            run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

            user_provided_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)

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
                ModelSchema.TRAINING_DATASET_ID_KEY,
            ) as training_dataset_id, upload_and_update_dataset(
                dr_client,
                holdout_dataset_filepath,
                model_metadata_yaml_file,
                ModelSchema.HOLDOUT_DATASET_ID_KEY,
            ) as holdout_dataset_id:
                try:
                    git_repo.git.add(model_metadata_yaml_file)
                    git_repo.git.commit("-m", "Update training / holdout dataset(s)")

                    printout("Run custom inference models GitHub action ...")
                    run_github_action(
                        workspace_path, git_repo, main_branch_name, "push", is_deploy=False
                    )

                    # Validation
                    custom_model = dr_client.fetch_custom_model_by_git_id(user_provided_id)
                    assert (
                        custom_model["externalMlopsStatsConfig"]["trainingDatasetId"]
                        == training_dataset_id
                    )
                    assert (
                        custom_model["externalMlopsStatsConfig"]["holdoutDatasetId"]
                        == holdout_dataset_id
                    )
                finally:
                    cleanup_models(dr_client, workspace_path)

        printout("Done")

    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_update_model_settings(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
    ):
        """
        An end-to-end case to test changes in deployment settings by the custom inference
        model GitHub action.
        """

        user_provided_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)

        # 1. Create a model just as a preliminary requirement (use GitHub action)
        printout(
            "Create a custom model as a preliminary requirement. "
            "Run custom model GitHub action (push event) ..."
        )
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        unique_string = unique_str()
        for settings_key, desired_settings_value in [
            (ModelSchema.NAME_KEY, f"Some another name {unique_string}"),
            (ModelSchema.DESCRIPTION_KEY, f"Some unique desc {unique_string}"),
            (ModelSchema.LANGUAGE_KEY, "Legacy"),
            (ModelSchema.TARGET_NAME_KEY, "XBH/AB_jr"),  # Taken from the associated dataset
            (ModelSchema.PREDICTION_THRESHOLD_KEY, 0.2),  # Assuming the model type is regression
        ]:
            custom_model = dr_client.fetch_custom_model_by_git_id(user_provided_id)
            actual_settings_value = custom_model[DrClient.MODEL_SETTINGS_KEYS_MAP[settings_key]]
            assert desired_settings_value != actual_settings_value, (
                f"Desired settings value '{desired_settings_value}' should be differ than the "
                f"actual '{actual_settings_value}'."
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
                    workspace_path, git_repo, main_branch_name, "push", is_deploy=False
                )

                # Validate
                custom_model = dr_client.fetch_custom_model_by_git_id(user_provided_id)
                actual_settings_value = custom_model[DrClient.MODEL_SETTINGS_KEYS_MAP[settings_key]]
                assert desired_settings_value == actual_settings_value, (
                    f"Desired settings value '{desired_settings_value}' should be equal to the "
                    f"actual '{actual_settings_value}'."
                )

    @pytest.fixture
    def dependency_package_name(self):
        """A fixture to return the dependency package name that is used within a functional test."""

        return "requests"

    @pytest.fixture
    def requirements_txt(self, dependency_package_name, model_metadata_yaml_file):
        """A fixture to create a requirements.txt file with a dependency."""

        requests_filepath = model_metadata_yaml_file.parent / "requirements.txt"
        with open(requests_filepath, "w", encoding="utf-8") as file:
            file.write(f"{dependency_package_name} >= 2.27.0\n")
        yield requests_filepath
        os.remove(requests_filepath)

    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_model_version_with_dependency(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata,
        main_branch_name,
        dependency_package_name,
        requirements_txt,
    ):
        """
        An end-to-end case to test model version with dependencies (requirements.txt).
        """

        shutil.rmtree(workspace_path / "models" / "model_2")
        user_provided_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)

        # 1. Create a model just as a preliminary requirement (use GitHub action)
        printout(
            "Create a custom model with dependency environment. "
            "Run custom model GitHub action (push event) ..."
        )
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        cm_version = dr_client.fetch_custom_model_latest_version_by_user_provided_id(
            user_provided_id
        )
        assert cm_version["dependencies"][0]["packageName"] == dependency_package_name
        with open(requirements_txt, "r", encoding="utf-8") as file:
            assert file.readline().strip() == cm_version["dependencies"][0]["line"]

        build_info = dr_client.get_custom_model_version_dependency_build_info(cm_version)
        assert build_info["buildStatus"] == "success"
