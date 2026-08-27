#  Copyright (c) 2023. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=too-many-arguments

"""
Functional tests for the custom inference model training/holdout data, which are handled by the
GitHub action. The functional tests are executed against a running DataRobot application. If
DataRobot is not accessible, the functional tests are skipped.
"""

import contextlib
from pathlib import Path

import pytest

from common.exceptions import DataRobotClientError
from schema_validator import ModelSchema
from tests.functional.conftest import cleanup_models
from tests.functional.conftest import printout
from tests.functional.conftest import run_github_action
from tests.functional.conftest import temporarily_replace_schema_value
from tests.functional.conftest import (
    temporarily_upload_training_dataset_for_structured_model,
)
from tests.functional.conftest import upload_and_update_dataset
from tests.functional.conftest import webserver_accessible


@pytest.mark.skipif(not webserver_accessible(), reason="DataRobot webserver is not accessible.")
@pytest.mark.usefixtures("cleanup", "skip_model_testing", "github_output")
class TestModelTrainingHoldoutData:
    """A test class that contains training and holdout data related test cases."""

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

        user_provided_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)
        with temporarily_upload_training_dataset_for_structured_model(
            dr_client, model_metadata_yaml_file, is_model_level=True, event_name="push"
        ) as (training_dataset_id_1, partition_column_1):
            try:
                self._commit_run_and_validata_training_holdout_data_for_model(
                    git_repo,
                    dr_client,
                    model_metadata_yaml_file,
                    workspace_path,
                    main_branch_name,
                    user_provided_id,
                    training_dataset_id_1,
                    partition_column=partition_column_1,
                )
            except Exception:
                cleanup_models(dr_client, workspace_path)
                raise

            with temporarily_upload_training_dataset_for_structured_model(
                dr_client, model_metadata_yaml_file, is_model_level=True, event_name="push"
            ) as (training_dataset_id_2, partition_column_2):
                try:
                    self._commit_run_and_validata_training_holdout_data_for_model(
                        git_repo,
                        dr_client,
                        model_metadata_yaml_file,
                        workspace_path,
                        main_branch_name,
                        user_provided_id,
                        training_dataset_id_2,
                        partition_column=partition_column_2,
                    )
                    assert training_dataset_id_1 != training_dataset_id_2
                    assert partition_column_1 == partition_column_2
                finally:
                    cleanup_models(dr_client, workspace_path)
        printout("Done")

    @staticmethod
    def _commit_run_and_validata_training_holdout_data_for_model(
        git_repo,
        dr_client,
        model_metadata_yaml_file,
        workspace_path,
        main_branch_name,
        user_provided_id,
        training_dataset_id,
        holdout_dataset_id=None,
        partition_column=None,
    ):
        git_repo.git.add(model_metadata_yaml_file)
        git_repo.git.commit("-m", "Update training / holdout dataset(s)")

        printout("Run custom inference models GitHub action ...")
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        # Validation
        custom_model = dr_client.fetch_custom_model_by_git_id(user_provided_id)
        if partition_column is not None:
            # Implicitly structured models
            assert custom_model["trainingDatasetId"] == training_dataset_id
            assert custom_model["trainingDataPartitionColumn"] == partition_column
        else:
            assert holdout_dataset_id is not None
            assert (
                custom_model["externalMlopsStatsConfig"]["trainingDatasetId"] == training_dataset_id
            )
            assert (
                custom_model["externalMlopsStatsConfig"]["holdoutDatasetId"] == holdout_dataset_id
            )
        versions = dr_client.fetch_custom_model_versions(custom_model["id"])
        assert len(versions) == 1

    def test_e2e_set_training_and_holdout_datasets_for_structured_model_version(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        main_branch_name,
    ):
        """
        Test a training dataset assignment for structured model version, by the custom inference
        model GitHub action. The training dataset contains a holdout column.
        """

        # 1. Create a model just as a preliminary requirement (use GitHub action)
        printout(
            "Create a custom model as a preliminary requirement. "
            "Run custom model GitHub action (push event) ..."
        )
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        with temporarily_upload_training_dataset_for_structured_model(
            dr_client, model_metadata_yaml_file, is_model_level=False, event_name="push"
        ) as (training_dataset_id_1, partition_column_1):
            try:
                self._commit_training_holdout_data_changes_for_model_version(
                    git_repo, model_metadata_yaml_file, workspace_path, main_branch_name
                )
            except Exception:
                cleanup_models(dr_client, workspace_path)
                raise

            with temporarily_upload_training_dataset_for_structured_model(
                dr_client, model_metadata_yaml_file, is_model_level=False, event_name="push"
            ) as (training_dataset_id_2, partition_column_2):
                try:
                    self._commit_training_holdout_data_changes_for_model_version(
                        git_repo, model_metadata_yaml_file, workspace_path, main_branch_name
                    )
                    self._validate_training_holdout_data_in_model_versions(
                        dr_client,
                        model_metadata,
                        training_dataset_id_1,
                        training_dataset_id_2,
                        partition_column_1=partition_column_1,
                        partition_column_2=partition_column_2,
                    )
                finally:
                    cleanup_models(dr_client, workspace_path)
        printout("Done")

    @staticmethod
    def _commit_training_holdout_data_changes_for_model_version(
        git_repo, model_metadata_yaml_file, workspace_path, main_branch_name
    ):
        git_repo.git.add(model_metadata_yaml_file)
        git_repo.git.commit("-m", "Update training / holdout dataset(s)")

        printout("Run custom inference models GitHub action ...")
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

    @staticmethod
    def _validate_training_holdout_data_in_model_versions(
        dr_client,
        model_metadata,
        training_dataset_id_1,
        training_dataset_id_2,
        holdout_dataset_id_1=None,
        partition_column_1=None,
        holdout_dataset_id_2=None,
        partition_column_2=None,
    ):
        assert training_dataset_id_1 != training_dataset_id_2

        custom_model = dr_client.fetch_custom_model_by_git_id(
            ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)
        )
        versions = dr_client.fetch_custom_model_versions(custom_model["id"])
        assert len(versions) == 3
        assert custom_model["latestVersion"]["id"] == versions[0]["id"]
        assert versions[0]["trainingData"]["datasetId"] == training_dataset_id_2
        assert versions[1]["trainingData"]["datasetId"] == training_dataset_id_1
        training_data = versions[2].get("trainingData")
        assert training_data is None or training_data.get("datasetId") is None
        holdout_data = versions[2].get("holdoutData")
        if holdout_dataset_id_1:
            assert holdout_dataset_id_2 is not None
            assert holdout_dataset_id_1 != holdout_dataset_id_2
            assert versions[0]["holdoutData"]["datasetId"] == holdout_dataset_id_2
            assert versions[1]["holdoutData"]["datasetId"] == holdout_dataset_id_1
            assert holdout_data is None or holdout_data.get("datasetId") is None
        else:
            assert partition_column_1 is not None
            assert partition_column_1 == partition_column_2
            assert versions[0]["holdoutData"]["partitionColumn"] == partition_column_2
            assert versions[1]["holdoutData"]["partitionColumn"] == partition_column_1
            assert holdout_data is None or holdout_data.get("partitionColumn") is None

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
            dr_client, model_metadata_yaml_file, is_model_level=True, event_name="push"
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
                    )

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
            new_value=ModelSchema.TARGET_TYPE_UNSTRUCTURED_OTHER,
        ):
            # 1. Create a model just as a preliminary requirement (use GitHub action)
            printout(
                "Create a custom model as a preliminary requirement. "
                "Run custom model GitHub action (push event) ..."
            )
            run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

            user_provided_id = ModelSchema.get_value(model_metadata, ModelSchema.MODEL_ID_KEY)

            with self._set_and_upload_training_and_holdout_data(
                1, dr_client, model_metadata_yaml_file, is_model_level=True
            ) as (training_dataset_id_1, holdout_dataset_id_1):
                try:
                    self._commit_run_and_validata_training_holdout_data_for_model(
                        git_repo,
                        dr_client,
                        model_metadata_yaml_file,
                        workspace_path,
                        main_branch_name,
                        user_provided_id,
                        training_dataset_id_1,
                        holdout_dataset_id=holdout_dataset_id_1,
                    )
                except Exception:
                    cleanup_models(dr_client, workspace_path)
                    raise

                with self._set_and_upload_training_and_holdout_data(
                    2, dr_client, model_metadata_yaml_file, is_model_level=True
                ) as (training_dataset_id_2, holdout_dataset_id_2):
                    try:
                        self._commit_run_and_validata_training_holdout_data_for_model(
                            git_repo,
                            dr_client,
                            model_metadata_yaml_file,
                            workspace_path,
                            main_branch_name,
                            user_provided_id,
                            training_dataset_id_2,
                            holdout_dataset_id=holdout_dataset_id_2,
                        )
                        assert training_dataset_id_1 != training_dataset_id_2
                        assert holdout_dataset_id_1 != holdout_dataset_id_2
                    finally:
                        cleanup_models(dr_client, workspace_path)

        printout("Done")

    @contextlib.contextmanager
    def _set_and_upload_training_and_holdout_data(
        self, index, dr_client, model_metadata_yaml_file, is_model_level
    ):
        printout(
            f"({index}) Upload training and holdout datasets for unstructured model. "
            f"is_model_level: {is_model_level}"
        )
        datasets_root = Path(__file__).parent / ".." / "datasets"
        training_dataset_filepath = (
            datasets_root / "juniors_3_year_stats_regression_unstructured_training.csv"
        )
        holdout_dataset_filepath = (
            datasets_root / "juniors_3_year_stats_regression_unstructured_holdout.csv"
        )
        section_key = (
            ModelSchema.SETTINGS_SECTION_KEY if is_model_level else ModelSchema.VERSION_KEY
        )
        with upload_and_update_dataset(
            dr_client,
            training_dataset_filepath,
            model_metadata_yaml_file,
            section_key,
            ModelSchema.TRAINING_DATASET_ID_KEY,
        ) as training_dataset_id, upload_and_update_dataset(
            dr_client,
            holdout_dataset_filepath,
            model_metadata_yaml_file,
            section_key,
            ModelSchema.HOLDOUT_DATASET_ID_KEY,
        ) as holdout_dataset_id:
            yield training_dataset_id, holdout_dataset_id

    def test_e2e_set_training_and_holdout_datasets_for_unstructured_model_version(
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
            new_value=ModelSchema.TARGET_TYPE_UNSTRUCTURED_OTHER,
        ):
            # 1. Create a model just as a preliminary requirement (use GitHub action)
            printout(
                "Create a custom model as a preliminary requirement. "
                "Run custom model GitHub action (push event) ..."
            )
            run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

            with self._set_and_upload_training_and_holdout_data(
                1, dr_client, model_metadata_yaml_file, is_model_level=False
            ) as (training_dataset_id_1, holdout_dataset_id_1):
                try:
                    self._commit_training_holdout_data_changes_for_model_version(
                        git_repo, model_metadata_yaml_file, workspace_path, main_branch_name
                    )
                except Exception:
                    cleanup_models(dr_client, workspace_path)
                    raise

                with self._set_and_upload_training_and_holdout_data(
                    2, dr_client, model_metadata_yaml_file, is_model_level=False
                ) as (training_dataset_id_2, holdout_dataset_id_2):
                    try:
                        self._commit_training_holdout_data_changes_for_model_version(
                            git_repo, model_metadata_yaml_file, workspace_path, main_branch_name
                        )
                        self._validate_training_holdout_data_in_model_versions(
                            dr_client,
                            model_metadata,
                            training_dataset_id_1,
                            training_dataset_id_2,
                            holdout_dataset_id_1=holdout_dataset_id_1,
                            holdout_dataset_id_2=holdout_dataset_id_2,
                        )
                    finally:
                        cleanup_models(dr_client, workspace_path)

        printout("Done")
