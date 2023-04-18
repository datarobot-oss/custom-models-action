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

import pytest
import yaml

from common.convertors import MemoryConvertor
from common.github_env import GitHubEnv
from dr_client import DrClient
from schema_validator import ModelSchema
from tests.conftest import unique_str
from tests.functional.conftest import NUMBER_OF_MODELS_IN_TEST
from tests.functional.conftest import increase_model_memory_by_1mb
from tests.functional.conftest import printout
from tests.functional.conftest import run_github_action
from tests.functional.conftest import temporarily_replace_schema_value
from tests.functional.conftest import webserver_accessible


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

        printout("Create models as a prerequisite for this test ...")
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

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
    def test_e2e_push_event_with_multiple_changes_and_a_frozen_version_in_between(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
    ):
        """
        An end-to-end case to test multiple commits, each is followed by a push event. In between,
        a model's package will be created, which means the associated version will be frozen.
        Consequently, the followed version will be a major update.,
        """

        user_provided_id = model_metadata[ModelSchema.MODEL_ID_KEY]
        # 1. Make three changes, one at a time on the main branch
        printout(
            "Make 3 changes one at a time on the main branch ... "
            f"user_provided_id: {user_provided_id}"
        )
        prev_latest_version = None
        sequence_of_create_package_flags = [False, False, True, False, False]
        for index, create_package in enumerate(sequence_of_create_package_flags):
            # 2. Make a change and commit it
            printout(f"Increase memory ... {index + 1}")
            new_memory = increase_model_memory_by_1mb(model_metadata_yaml_file)
            git_repo.git.add(model_metadata_yaml_file)
            git_repo.git.commit("-m", f"Increase memory to {new_memory}")

            # 3. Run GitHub pull request action
            printout("Run custom model GitHub action (push event) ...")
            run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

            latest_version = dr_client.fetch_custom_model_latest_version_by_user_provided_id(
                user_provided_id
            )

            was_package_created_in_prev_iteration = bool(
                index and sequence_of_create_package_flags[index - 1]
            )
            self._validate_major_minor_model_version(
                prev_latest_version, latest_version, was_package_created_in_prev_iteration
            )
            prev_latest_version = latest_version

            if create_package:
                dr_client.create_model_package_from_custom_model_version(latest_version["id"])
        printout("Done")

    @staticmethod
    def _validate_major_minor_model_version(
        prev_latest_version, latest_version, was_package_created_in_prev_iteration
    ):
        if prev_latest_version:
            if was_package_created_in_prev_iteration:
                assert latest_version["versionMajor"] == prev_latest_version["versionMajor"] + 1
                assert latest_version["versionMinor"] == 0
            else:
                assert latest_version["versionMajor"] == prev_latest_version["versionMajor"]
                assert latest_version["versionMinor"] == prev_latest_version["versionMinor"] + 1
        else:
            assert latest_version["versionMajor"] == 1
            assert latest_version["versionMinor"] == 0

    def test_is_accessible(self):
        """A test case to check whether DataRobot webserver is accessible."""

        assert webserver_accessible()

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

    # pylint: disable=too-many-locals
    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_no_new_model_version_upon_pr_with_unrelated_changes(
        self,
        dr_client,
        workspace_path,
        git_repo,
        numbered_model_metadata,
        numbered_model_metadata_yaml_file,
        main_branch_name,
        feature_branch_name,
        merge_branch_name,
    ):
        """
        This case validates that no new model's version is created by a pull-request that affects
        another unrelated model. This can happen if the last change for the first model was made
        to its settings, before the pull-request of the other model was created.
        """

        # This test requires two models
        assert NUMBER_OF_MODELS_IN_TEST == 2

        # 1. Create two models as a preliminary requirement (use GitHub action)
        printout(
            "Create 2 custom models as a preliminary requirement. "
            "Run custom model GitHub action (push event) ..."
        )
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        model_1_metadata = numbered_model_metadata(model_number=1)
        model_1_metadata_yaml_file = numbered_model_metadata_yaml_file(model_number=1)
        model_2_metadata = numbered_model_metadata(model_number=2)
        model_2_metadata_yaml_file = numbered_model_metadata_yaml_file(model_number=2)

        # 2. Retrieve the models from DataRobot
        model_1_user_provided_id = model_1_metadata[ModelSchema.MODEL_ID_KEY]
        dr_model_1 = dr_client.fetch_custom_model_by_git_id(model_1_user_provided_id)
        custom_model_1_name = dr_model_1[DrClient.MODEL_SETTINGS_KEYS_MAP[ModelSchema.NAME_KEY]]
        custom_model_1_origin_version_id = dr_model_1["latestVersion"]["id"]

        model_2_user_provided_id = model_2_metadata[ModelSchema.MODEL_ID_KEY]
        dr_model_2 = dr_client.fetch_custom_model_by_git_id(model_2_user_provided_id)
        custom_model_2_origin_version_id = dr_model_2["latestVersion"]["id"]

        # 3. Apply settings change (rename label) to the first model
        model_1_metadata[ModelSchema.SETTINGS_SECTION_KEY][
            ModelSchema.NAME_KEY
        ] = f"{custom_model_1_name}-new"
        with open(model_1_metadata_yaml_file, "w", encoding="utf-8") as fd:
            yaml.safe_dump(model_1_metadata, fd)

        git_repo.git.add(model_1_metadata_yaml_file)
        git_repo.git.commit("-m", "Update the name of the first model.")

        printout("Run custom inference models GitHub action (push) ...")
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        self._validate_custom_model_action_metrics(
            expected_num_affected_models=1,
            expected_num_updated_settings=1,
            expected_num_created_versions=0,
        )

        # 4. Validate that no new versions were created in both models
        dr_model_1 = dr_client.fetch_custom_model_by_git_id(model_1_user_provided_id)
        assert custom_model_1_origin_version_id == dr_model_1["latestVersion"]["id"]
        dr_model_2 = dr_client.fetch_custom_model_by_git_id(model_2_user_provided_id)
        assert custom_model_2_origin_version_id == dr_model_2["latestVersion"]["id"]

        # 5. Create a feature-branch
        printout("Create a feature branch and checkout ...")
        feature_branch = git_repo.create_head(feature_branch_name)

        # 6. Make a memory change to the second model and merge back.
        checks = [self._increase_memory_check]
        self._run_checks(
            checks,
            feature_branch,
            git_repo,
            workspace_path,
            main_branch_name,
            model_2_metadata,
            model_2_metadata_yaml_file,
            merge_branch_name,
            dr_client,
        )

        self._validate_custom_model_action_metrics(
            expected_num_affected_models=1,
            expected_num_updated_settings=1,
            expected_num_created_versions=1,
        )

        # 4. Validate that a new version was only created to the second model
        dr_model_1 = dr_client.fetch_custom_model_by_git_id(model_1_user_provided_id)
        assert custom_model_1_origin_version_id == dr_model_1["latestVersion"]["id"]
        dr_model_2 = dr_client.fetch_custom_model_by_git_id(model_2_user_provided_id)
        assert custom_model_2_origin_version_id != dr_model_2["latestVersion"]["id"]

    @staticmethod
    def _validate_custom_model_action_metrics(
        expected_num_affected_models, expected_num_updated_settings, expected_num_created_versions
    ):
        github_output_filepath = GitHubEnv.github_output()
        with open(github_output_filepath, encoding="utf-8") as file:
            lines = file.readlines()

        index = 0
        actual_num_affected_models = None
        actual_num_updated_settings = None
        actual_num_created_versions = None
        for line in reversed(lines):
            key, value = line.strip().split("=")
            if key.startswith("models--total-affected"):
                actual_num_affected_models = int(value)
                index += 1
            elif key.startswith("models--total-updated-settings"):
                actual_num_updated_settings = int(value)
                index += 1
            elif key.startswith("models--total-created-versions"):
                actual_num_created_versions = int(value)
                index += 1

            if index == 3:
                break

        assert actual_num_affected_models == expected_num_affected_models
        assert actual_num_updated_settings == expected_num_updated_settings
        assert actual_num_created_versions == expected_num_created_versions

    @staticmethod
    def _print_custom_model_action_metrics():
        github_output_filepath = GitHubEnv.github_output()
        with open(github_output_filepath, encoding="utf-8") as file:
            print(file.read())

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
