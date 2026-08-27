#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=too-many-arguments

"""
Functional tests for the deployment GitHub action. Functional tests are executed against a running
DataRobot application. If DataRobot is not accessible, the functional tests are skipped.
"""

import contextlib
import logging
import os
import re
from pathlib import Path

import pytest
import yaml

from common import constants
from common.exceptions import AssociatedModelNotFound
from common.exceptions import DataRobotClientError
from common.exceptions import IllegalModelDeletion
from common.namepsace import Namespace
from deployment_info import DeploymentInfo
from metrics import Metrics
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from tests.conftest import unique_str
from tests.functional.conftest import cleanup_models
from tests.functional.conftest import increase_model_memory_by_1mb
from tests.functional.conftest import printout
from tests.functional.conftest import replace_and_store_schema
from tests.functional.conftest import run_github_action
from tests.functional.conftest import save_new_metadata_and_commit
from tests.functional.conftest import temporarily_replace_schema
from tests.functional.conftest import (
    temporarily_upload_training_dataset_for_structured_model,
)
from tests.functional.conftest import upload_and_update_dataset
from tests.functional.conftest import webserver_accessible


@pytest.fixture(name="deployment_metadata")
def fixture_deployment_metadata(deployment_metadata_yaml_file):
    """A fixture to load and return a deployment metadata from a given yaml file definition."""

    with open(deployment_metadata_yaml_file, encoding="utf-8") as fd:
        raw_metadata = yaml.safe_load(fd)
        return DeploymentSchema.validate_and_transform_single(raw_metadata)


@pytest.fixture(name="skip_association")
def fixture_skip_association(git_repo, deployment_metadata_yaml_file, deployment_metadata):
    """A fixture to remove the association section from a given deployment metadata."""

    deployment_metadata[DeploymentSchema.SETTINGS_SECTION_KEY].pop(DeploymentSchema.ASSOCIATION_KEY)
    save_new_metadata_and_commit(
        deployment_metadata, deployment_metadata_yaml_file, git_repo, "Skip association"
    )


@pytest.fixture(name="cleanup")
def fixture_cleanup(dr_client, workspace_path, deployment_metadata):
    """A fixture to delete all deployments and models that were created from the source tree."""

    yield

    cleanup_deployment(dr_client, deployment_metadata)
    # NOTE: we have more than one model in the tree
    cleanup_models(dr_client, workspace_path)


def cleanup_deployment(dr_client, deployment_metadata):
    """Silently delete a deployment that is specified in a deployment's metadata."""

    try:
        dr_client.delete_deployment_by_git_id(
            deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        )
    except (IllegalModelDeletion, DataRobotClientError):
        pass


@pytest.mark.skipif(not webserver_accessible(), reason="DataRobot webserver is not accessible.")
@pytest.mark.usefixtures("cleanup", "build_repo_for_testing", "github_output")
class TestDeploymentGitHubActions:
    """Contains cases to test the deployment GitHub action."""

    @contextlib.contextmanager
    def _upload_actuals_dataset(
        self, dr_client, deployment_metadata, deployment_metadata_yaml_file
    ):
        association_id_column = DeploymentSchema.get_value(
            deployment_metadata,
            DeploymentSchema.SETTINGS_SECTION_KEY,
            DeploymentSchema.ASSOCIATION_KEY,
            DeploymentSchema.ASSOCIATION_ASSOCIATION_ID_COLUMN_KEY,
        )
        if association_id_column:
            actuals_filepath = (
                Path(__file__).parent
                / ".."
                / "datasets"
                / "juniors_3_year_stats_regression_actuals.csv"
            )
            with upload_and_update_dataset(
                dr_client,
                actuals_filepath,
                deployment_metadata_yaml_file,
                DeploymentSchema.SETTINGS_SECTION_KEY,
                DeploymentSchema.ASSOCIATION_KEY,
                DeploymentSchema.ASSOCIATION_ACTUALS_DATASET_ID_KEY,
            ) as dataset_id:
                yield dataset_id

    @staticmethod
    def _commit_changes_by_event(commit_message, event_name, git_repo):
        """
        For 'push' event the changes are committed into the master using a single commit. For
        'pull_request' event, the changes are committed into a feature branch and then a merge
        branch is created.
        """

        os.chdir(git_repo.working_dir)
        if event_name == "push":
            git_repo.git.commit("-a", "-m", commit_message)
        elif event_name == "pull_request":
            feature_branch_name = "feature-branch"
            if feature_branch_name in git_repo.heads:
                feature_branch = git_repo.heads[feature_branch_name]
            else:
                feature_branch = git_repo.create_head(feature_branch_name)
            feature_branch.checkout()

            git_repo.git.add("--all")
            git_repo.git.commit("-m", commit_message, "--no-verify")

            merge_branch_name = "merge-branch"
            if merge_branch_name in git_repo.heads:
                # Delete the merge branch to enable creation of another merge branch
                git_repo.delete_head(merge_branch_name, "--force")

            merge_branch = git_repo.create_head(merge_branch_name, "master")
            git_repo.head.reference = merge_branch
            git_repo.head.reset(index=True, working_tree=True)
            git_repo.git.merge(feature_branch, "--no-ff")
        else:
            raise Exception("Unsupported git event!")

    @pytest.mark.parametrize("event_name", ["push", "pull_request"])
    @pytest.mark.usefixtures("skip_model_testing")
    def test_e2e_deployment_create(
        self,
        dr_client,
        workspace_path,
        git_repo,
        deployment_metadata,
        deployment_metadata_yaml_file,
        main_branch_name,
        event_name,
    ):
        """An end-to-end case to test a deployment creation."""

        # Upload actuals dataset and set the deployment metadata with the dataset ID
        printout("Upload actuals dataset")
        with self._upload_actuals_dataset(
            dr_client, deployment_metadata, deployment_metadata_yaml_file
        ):
            self._commit_changes_by_event("Update actuals dataset", event_name, git_repo)

            printout("Run the GitHub action to create a model and deployment")
            run_github_action(
                workspace_path, git_repo, main_branch_name, event_name, is_deploy=True
            )

        # 4. Validate
        printout("Validate ...")
        local_user_provided_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        if event_name == "push":
            assert dr_client.fetch_deployment_by_git_id(local_user_provided_id) is not None
        elif event_name == "pull_request":
            assert dr_client.fetch_deployment_by_git_id(local_user_provided_id) is None
        else:
            assert False, f"Unsupported GitHub event name: {event_name}"

        printout("Done")

    @pytest.mark.usefixtures("skip_model_testing")
    def test_e2e_existing_deployment_with_deleted_model_metadata(
        self,
        dr_client,
        workspace_path,
        git_repo,
        deployment_metadata,
        deployment_metadata_yaml_file,
        main_branch_name,
        feature_branch_name,
        merge_branch_name,
        model_metadata_yaml_file,
    ):
        """
        An end-to-end case to test a scenario in which the user deletes a model metadata
        definition in a PR while the model is deployed in DataRobot.
        """

        # Upload actuals dataset and set the deployment metadata with the dataset ID
        printout("Upload actuals dataset")
        with self._upload_actuals_dataset(
            dr_client, deployment_metadata, deployment_metadata_yaml_file
        ):
            self._commit_changes_by_event("Update actuals dataset", "push", git_repo)

            printout("Run the GitHub action to create a model and deployment")
            run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=True)

            # Validate that the deployment was created in DataRobot
            local_user_provided_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
            assert dr_client.fetch_deployment_by_git_id(local_user_provided_id) is not None

            # Create a feature branch and checkout
            printout("Create a feature branch ...")
            feature_branch = git_repo.create_head(feature_branch_name)
            feature_branch.checkout()

            # Delete the local model definition (metadata) YAML file
            os.remove(model_metadata_yaml_file)
            git_repo.git.add(model_metadata_yaml_file)
            git_repo.git.commit("-m", "Removed model's definition")

            # Create a merge branch
            merge_branch = git_repo.create_head(merge_branch_name, main_branch_name)
            git_repo.head.reference = merge_branch
            git_repo.head.reset(index=True, working_tree=True)
            git_repo.git.merge(feature_branch, "--no-ff")

            # Run the action and expect for exception
            printout(
                "Run the GitHub action in a feature branch and expect for"
                "AssociatedModelNotFound exception"
            )
            with pytest.raises(AssociatedModelNotFound):
                run_github_action(
                    workspace_path, git_repo, main_branch_name, "pull_request", is_deploy=True
                )
        printout("Done")

    @pytest.mark.parametrize("event_name", ["push", "pull_request"])
    @pytest.mark.usefixtures("set_model_dataset_for_testing", "set_deployment_actuals_dataset")
    def test_e2e_deployment_model_replacement(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata_yaml_file,
        deployment_metadata,
        deployment_metadata_yaml_file,
        main_branch_name,
        event_name,
    ):
        """An end-to-end case to test a model replacement in a deployment."""

        # Disable challengers
        printout("Disable challengers ...")
        self._enable_challenger(git_repo, deployment_metadata, deployment_metadata_yaml_file, False)

        (
            _,
            latest_deployment_model_version_id,
            latest_model_version,
        ) = self._deploy_a_model_than_run_github_action_to_replace_or_challenge(
            dr_client,
            git_repo,
            workspace_path,
            main_branch_name,
            deployment_metadata,
            model_metadata_yaml_file,
            event_name,
        )

        if event_name == "push":
            assert latest_deployment_model_version_id
            assert latest_deployment_model_version_id == latest_model_version["id"]
        elif event_name == "pull_request":
            assert latest_deployment_model_version_id
            assert latest_model_version["id"]
            assert latest_deployment_model_version_id != latest_model_version["id"]
        else:
            assert False, f"Unsupported GitHub event name: {event_name}"
        printout("Done")

    @classmethod
    def _deploy_a_model_than_run_github_action_to_replace_or_challenge(
        cls,
        dr_client,
        git_repo,
        workspace_path,
        main_branch_name,
        deployment_metadata,
        model_metadata_yaml_file,
        event_name,
    ):
        # Create a deployment
        printout("Run the GitHub action (push event) to create a model and deployment")
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=True)
        local_user_provided_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        deployments = dr_client.fetch_deployments()
        assert any(d.get("userProvidedId") == local_user_provided_id for d in deployments)

        # Make a local change to the model and commit
        printout("Make a change to the model and run custom model GitHub action (push event) ...")
        new_memory = increase_model_memory_by_1mb(
            git_repo, model_metadata_yaml_file, do_commit=False
        )

        # Set the reason for the replacement
        replace_and_store_schema(
            model_metadata_yaml_file,
            ModelSchema.VERSION_KEY,
            ModelSchema.MODEL_REPLACEMENT_REASON_KEY,
            metadata_or_value=ModelSchema.MODEL_REPLACEMENT_REASON_ERRORS,
        )

        cls._commit_changes_by_event(
            f"Increase memory to {new_memory} and update the reason for replacement",
            event_name,
            git_repo,
        )

        # Run the GitHub action to replace the latest model in a deployment
        printout(f"Run the GitHub action ({event_name} event)")
        run_github_action(workspace_path, git_repo, main_branch_name, event_name, is_deploy=True)

        printout("Validate ...")
        deployments = dr_client.fetch_deployments()
        the_deployment = next(
            d for d in deployments if d.get("userProvidedId") == local_user_provided_id
        )

        latest_deployment_model_version_id = the_deployment["model"]["customModelImage"][
            "customModelVersionId"
        ]
        local_user_provided_id = deployment_metadata[DeploymentSchema.MODEL_ID_KEY]
        latest_model_version = dr_client.fetch_custom_model_latest_version_by_user_provided_id(
            local_user_provided_id
        )
        return the_deployment, latest_deployment_model_version_id, latest_model_version

    @staticmethod
    def _enable_challenger(
        git_repo, deployment_metadata, deployment_metadata_yaml_file, enabled=True
    ):
        settings = deployment_metadata.get(DeploymentSchema.SETTINGS_SECTION_KEY, {})
        settings[DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY] = enabled
        deployment_metadata[DeploymentSchema.SETTINGS_SECTION_KEY] = settings
        save_new_metadata_and_commit(
            deployment_metadata,
            deployment_metadata_yaml_file,
            git_repo,
            "Enable challenger",
        )

    @pytest.mark.usefixtures("skip_model_testing", "set_deployment_actuals_dataset")
    @pytest.mark.parametrize("github_event", ["push", "pull_request"])
    def test_e2e_deployment_delete(
        self,
        dr_client,
        workspace_path,
        git_repo,
        deployment_metadata,
        deployment_metadata_yaml_file,
        main_branch_name,
        github_event,
    ):
        """An end-to-end case to test a deployment deletion."""

        # Setup: create a model and deployment. Run the GitHub action.
        printout("Run the GitHub action (push event) to create a model and deployment")
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=True)
        deployments = dr_client.fetch_deployments()
        local_user_provided_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        assert any(d.get("userProvidedId") == local_user_provided_id for d in deployments)

        # Delete a deployment local definition yaml file
        printout(f"Run the GitHub action to delete deployment, github_event: {github_event}")
        # Note: do not use os.remove, because it'll eventually remove the whole deployments
        # directory in a feature branch.
        os.rename(deployment_metadata_yaml_file, f"{deployment_metadata_yaml_file}.deleted")
        self._commit_changes_by_event(
            "Delete the deployment definition file", github_event, git_repo
        )

        # Run the GitHub action but disallow deployment deletion
        run_github_action(
            workspace_path,
            git_repo,
            main_branch_name,
            event_name=github_event,
            is_deploy=True,
            allow_deployment_deletion=False,
        )
        printout("Validate ...")
        deployments = dr_client.fetch_deployments()
        local_user_provided_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        assert any(d.get("userProvidedId") == local_user_provided_id for d in deployments)

        # Run the GitHub action with allowed deployment deletion
        printout("Run the GitHub action with allowed deletion")
        run_github_action(
            workspace_path,
            git_repo,
            main_branch_name,
            github_event,
            is_deploy=True,
            allow_deployment_deletion=True,
        )
        printout("Validate ...")
        deployments = dr_client.fetch_deployments()
        local_user_provided_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        if github_event == "push":
            assert all(d.get("userProvidedId") != local_user_provided_id for d in deployments)
        elif github_event == "pull_request":
            assert any(d.get("userProvidedId") == local_user_provided_id for d in deployments)
        else:
            assert False, f"GitHub event '{github_event}' is not supported!"
        printout("Done")

    @pytest.mark.parametrize("event_name", ["push", "pull_request"])
    @pytest.mark.usefixtures("set_model_dataset_for_testing", "set_deployment_actuals_dataset")
    def test_e2e_deployment_model_challengers(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata_yaml_file,
        deployment_metadata,
        deployment_metadata_yaml_file,
        main_branch_name,
        event_name,
    ):
        """An end-to-end case to test challengers in a deployment."""

        # Enable challengers (although it is the default)
        printout("Enable challengers ...")
        self._enable_challenger(git_repo, deployment_metadata, deployment_metadata_yaml_file, True)

        (
            the_deployment,
            latest_deployment_model_version_id,
            latest_model_version,
        ) = self._deploy_a_model_than_run_github_action_to_replace_or_challenge(
            dr_client,
            git_repo,
            workspace_path,
            main_branch_name,
            deployment_metadata,
            model_metadata_yaml_file,
            event_name,
        )

        assert latest_deployment_model_version_id
        assert latest_model_version["id"]
        assert latest_deployment_model_version_id != latest_model_version["id"]

        challengers = dr_client.fetch_challengers(the_deployment["id"])
        if event_name == "push":
            assert len(challengers) == 2, challengers
            assert challengers[-1]["model"]["id"] == latest_model_version["id"]
        elif event_name == "pull_request":
            assert len(challengers) == 1, challengers
        else:
            assert False, f"Unsupported GitHub event name: {event_name}"
        printout("Done")

    @pytest.mark.parametrize("event_name", ["push", "pull_request"])
    @pytest.mark.usefixtures("skip_model_testing", "set_deployment_actuals_dataset")
    def test_e2e_deployment_settings(
        self,
        dr_client,
        workspace_path,
        git_repo,
        model_metadata_yaml_file,
        deployment_metadata,
        deployment_metadata_yaml_file,
        main_branch_name,
        event_name,
        github_output,
    ):
        """An end-to-end case to test changes in deployment settings."""

        with temporarily_upload_training_dataset_for_structured_model(
            dr_client, model_metadata_yaml_file, is_model_level=False, event_name=event_name
        ):
            try:
                # Run the GitHub action to create a model and deployment
                printout("Run the GitHub action (push event) to create a model and deployment")
                run_github_action(
                    workspace_path, git_repo, main_branch_name, "push", is_deploy=True
                )
                local_user_provided_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
                deployment = dr_client.fetch_deployment_by_git_id(local_user_provided_id)
                assert deployment is not None

                for check_func in [
                    self._test_deployment_entity_update,
                    self._test_deployment_settings,
                ]:
                    with check_func(
                        dr_client,
                        deployment,
                        deployment_metadata,
                        deployment_metadata_yaml_file,
                        event_name,
                        github_output,
                    ):
                        self._commit_changes_by_event(
                            f"Update settings by {check_func.__name__}", event_name, git_repo
                        )

                        self._reset_github_output_metrics(github_output)
                        printout(f"Run the GitHub action ({event_name}")
                        run_github_action(
                            workspace_path, git_repo, main_branch_name, event_name, is_deploy=True
                        )
                        self._validate_deployments_metric(
                            Metrics.total_affected, event_name, github_output
                        )
            finally:
                cleanup_deployment(dr_client, deployment_metadata)
                cleanup_models(dr_client, workspace_path)

    @staticmethod
    def _reset_github_output_metrics(github_output_filepath):
        github_output_filepath = Path(github_output_filepath)
        if github_output_filepath.is_file():
            github_output_filepath.unlink()
        with open(github_output_filepath, "w", encoding="utf-8"):
            pass

    @staticmethod
    def _validate_deployments_metric(metric, event_name, github_output_filepath):
        with open(github_output_filepath, "r", encoding="utf-8") as file:
            github_output_content = file.read()
        metric_label = Metrics.metric_label(constants.Label.DEPLOYMENTS, metric.label)
        pattern = f"{metric_label}=(.*)"
        items = re.findall(pattern, github_output_content)
        assert len(items) == 1, (
            "Unexpected occurrences number for "
            f"pattern '{pattern}' in content: {github_output_content}"
        )
        actual_value = int(items[0])
        expected_affected = 1 if event_name == "push" else 0
        assert expected_affected == actual_value

    @staticmethod
    @contextlib.contextmanager
    def _test_deployment_entity_update(
        dr_client,
        deployment,
        deployment_metadata,
        deployment_metadata_yaml_file,
        event_name,
        github_output=None,  # pylint: disable=unused-argument
    ):
        printout("Update deployment entity")

        origin_deployment = deployment
        deployment_info = DeploymentInfo(deployment_metadata_yaml_file, deployment_metadata)

        deployment_attrs = [
            (DeploymentSchema.LABEL_KEY, "label"),
            (DeploymentSchema.DESCRIPTION_KEY, "description"),
        ]
        for schema_attr, dr_attr in deployment_attrs:
            new_value = f"{origin_deployment[dr_attr]} - NEW"
            deployment_info.set_settings_value(schema_attr, value=new_value)

        with temporarily_replace_schema(deployment_metadata_yaml_file, deployment_info.metadata):
            yield

        new_deployment = dr_client.fetch_deployment_by_git_id(deployment_info.user_provided_id)

        if event_name == "push":
            for schema_attr, dr_attr in deployment_attrs:
                assert new_deployment[dr_attr] == deployment_info.get_settings_value(schema_attr)
        elif event_name == "pull_request":
            for _, dr_attr in deployment_attrs:
                assert new_deployment[dr_attr] == origin_deployment[dr_attr]
        else:
            assert False, f"Unsupported GitHub event name: {event_name}"

    @classmethod
    @contextlib.contextmanager
    def _test_deployment_settings(
        cls,
        dr_client,
        deployment,
        deployment_metadata,
        deployment_metadata_yaml_file,
        event_name,
        github_output,
    ):
        printout("Update deployment settings")

        deployment_info = DeploymentInfo(deployment_metadata_yaml_file, deployment_metadata)
        origin_deployment_settings = dr_client.fetch_deployment_settings(
            deployment["id"], deployment_info
        )

        deployment_info = cls._set_new_deployment_settings(
            origin_deployment_settings, deployment_info
        )

        with temporarily_replace_schema(deployment_metadata_yaml_file, deployment_info.metadata):
            yield

        cls._validate_deployment_settings(
            dr_client,
            deployment,
            deployment_info,
            origin_deployment_settings,
            event_name,
            github_output,
        )

    @staticmethod
    def _set_new_deployment_settings(origin_deployment_settings, deployment_info):
        new_value = not origin_deployment_settings["targetDrift"]["enabled"]
        deployment_info.set_settings_value(
            DeploymentSchema.ENABLE_TARGET_DRIFT_KEY, value=new_value
        )

        new_value = not origin_deployment_settings["featureDrift"]["enabled"]
        deployment_info.set_settings_value(
            DeploymentSchema.ENABLE_FEATURE_DRIFT_KEY, value=new_value
        )

        new_value = not origin_deployment_settings["segmentAnalysis"]["enabled"]
        deployment_info.set_settings_value(
            DeploymentSchema.SEGMENT_ANALYSIS_KEY,
            DeploymentSchema.ENABLE_SEGMENT_ANALYSIS_KEY,
            value=new_value,
        )

        new_value = not origin_deployment_settings["challengerModels"]["enabled"]
        deployment_info.set_settings_value(
            DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY, value=new_value
        )
        deployment_info.set_settings_value(
            DeploymentSchema.ENABLE_PREDICTIONS_COLLECTION_KEY, value=new_value
        )
        return deployment_info

    @classmethod
    def _validate_deployment_settings(
        cls,
        dr_client,
        deployment,
        deployment_info,
        origin_deployment_settings,
        event_name,
        github_output,
    ):
        new_deployment_settings = dr_client.fetch_deployment_settings(
            deployment["id"], deployment_info
        )
        expected_values = {}
        if event_name == "push":
            expected_values[
                DeploymentSchema.ENABLE_TARGET_DRIFT_KEY
            ] = deployment_info.get_settings_value(DeploymentSchema.ENABLE_TARGET_DRIFT_KEY)

            expected_values[
                DeploymentSchema.ENABLE_FEATURE_DRIFT_KEY
            ] = deployment_info.get_settings_value(DeploymentSchema.ENABLE_FEATURE_DRIFT_KEY)

            expected_values[
                DeploymentSchema.SEGMENT_ANALYSIS_KEY
            ] = deployment_info.get_settings_value(
                DeploymentSchema.SEGMENT_ANALYSIS_KEY,
                DeploymentSchema.ENABLE_SEGMENT_ANALYSIS_KEY,
            )
            expected_values[
                DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY
            ] = deployment_info.get_settings_value(DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY)
            expected_values[
                DeploymentSchema.ENABLE_PREDICTIONS_COLLECTION_KEY
            ] = deployment_info.get_settings_value(
                DeploymentSchema.ENABLE_PREDICTIONS_COLLECTION_KEY
            )
        elif event_name == "pull_request":
            expected_values[DeploymentSchema.ENABLE_TARGET_DRIFT_KEY] = origin_deployment_settings[
                "targetDrift"
            ]["enabled"]
            expected_values[DeploymentSchema.ENABLE_FEATURE_DRIFT_KEY] = origin_deployment_settings[
                "featureDrift"
            ]["enabled"]
            expected_values[DeploymentSchema.SEGMENT_ANALYSIS_KEY] = new_deployment_settings[
                "segmentAnalysis"
            ]["enabled"]
            expected_values[
                DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY
            ] = origin_deployment_settings["challengerModels"]["enabled"]
            expected_values[
                DeploymentSchema.ENABLE_PREDICTIONS_COLLECTION_KEY
            ] = origin_deployment_settings["predictionsDataCollection"]["enabled"]
        else:
            assert False, f"Unsupported GitHub event name: {event_name}"

        assert (
            expected_values[DeploymentSchema.ENABLE_TARGET_DRIFT_KEY]
            == new_deployment_settings["targetDrift"]["enabled"]
        )
        assert (
            expected_values[DeploymentSchema.ENABLE_FEATURE_DRIFT_KEY]
            == new_deployment_settings["featureDrift"]["enabled"]
        )
        assert (
            expected_values[DeploymentSchema.SEGMENT_ANALYSIS_KEY]
            == new_deployment_settings["segmentAnalysis"]["enabled"]
        )
        assert (
            expected_values[DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY]
            == new_deployment_settings["challengerModels"]["enabled"]
        )
        assert (
            expected_values[DeploymentSchema.ENABLE_PREDICTIONS_COLLECTION_KEY]
            == new_deployment_settings["predictionsDataCollection"]["enabled"]
        )
        cls._validate_deployments_metric(Metrics.total_updated_settings, event_name, github_output)

    @contextlib.contextmanager
    def _set_namespace(self, namespace):
        origin_namespace = Namespace.namespace()
        try:
            Namespace.uninit()
            Namespace.init(namespace)
            printout(f"Set new namespace, old: {origin_namespace}, new: {Namespace.namespace()}")
            yield
        finally:
            Namespace.uninit()
            Namespace.init(origin_namespace)

    @pytest.mark.usefixtures("skip_model_testing", "set_deployment_actuals_dataset")
    def test_deployment_and_model_fetch_for_different_namespaces(
        self, dr_client, workspace_path, git_repo, main_branch_name
    ):
        """An end-to-end case to test a deployment deletion."""

        # Create a model and deployment in the functional test namespace. Run the GitHub action.
        printout("Run the GitHub action (push event) to create a model and deployment")
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=True)

        # Validate deployments exist in namespace
        printout("Validate that models and deployments exist in the configured namespace.")
        deployments = dr_client.fetch_deployments()
        assert len(deployments) > 0

        custom_models = dr_client.fetch_custom_models()
        assert len(custom_models) > 0

        # Change namespace and verify zero fetched deployments
        new_namespace = f"datarobot/gh-functional-tests/non-existing-ns-{unique_str()}"
        with self._set_namespace(new_namespace):
            assert Namespace.namespace() == f"{new_namespace}/"

            printout("Validate non existing models nor deployments in the new namespace.")
            deployments = dr_client.fetch_deployments()
            assert len(deployments) == 0

            custom_models = dr_client.fetch_custom_models()
            assert len(custom_models) == 0

    @pytest.mark.usefixtures("skip_model_testing", "skip_association")
    def test_e2e_deployment_create_failure(
        self, workspace_path, git_repo, model_metadata_yaml_file, main_branch_name, caplog
    ):
        """
        An end-to-end case to test a failure of a background job during a deployment's creation.
        """

        printout("Run the GitHub action to create an erroneous model and deployment")
        with self._simulate_model_error(model_metadata_yaml_file):
            with pytest.raises(DataRobotClientError) as exec_info:
                with caplog.at_level(logging.WARNING):
                    run_github_action(
                        workspace_path, git_repo, main_branch_name, "push", is_deploy=True
                    )

            printout(str(exec_info.value))
            assert any(record.levelname in ("WARNING", "ERROR") for record in caplog.records)
            assert "WARNING" in str(exec_info.value) or "ERROR" in str(exec_info.value)
        printout("Done")

    @contextlib.contextmanager
    def _simulate_model_error(self, model_metadata_yaml_file):
        model_dir_path = Path(model_metadata_yaml_file).parent
        try:
            origin_pickle_file_path = next(model_dir_path.rglob("*.pkl"))
        except StopIteration:
            assert False, "Missing model's pickle artifact"
        tmp_pickle_file_path = origin_pickle_file_path.rename(f"{origin_pickle_file_path}.bak")
        yield
        tmp_pickle_file_path.rename(origin_pickle_file_path)
