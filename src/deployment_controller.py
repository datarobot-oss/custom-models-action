#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
This module controls and coordinate between local deployment definitions and DataRobot deployments.
In highlights, it scans and loads deployment definitions from the local source tree, perform
validations and then applies actions in DataRobot.
"""

import logging
from pathlib import Path

from common.data_types import DataRobotDeployment
from common.exceptions import AssociatedModelNotFound
from common.exceptions import AssociatedModelVersionNotFound
from common.exceptions import DataRobotClientError
from common.exceptions import DeploymentMetadataAlreadyExists
from common.exceptions import NoValidAncestor
from common.github_env import GitHubEnv
from deployment_info import DeploymentInfo
from model_controller import ControllerBase
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema

logger = logging.getLogger()


class DeploymentController(ControllerBase):
    """
    A deployment controller that coordinates between local deployment definitions to DataRobot
    deployments.
    """

    def __init__(self, options, model_controller, repo):
        super().__init__(options, repo)
        self._model_controller = model_controller
        self._deployments_info = {}
        self._datarobot_deployments = {}

    def _label(self):
        return self.DEPLOYMENTS_LABEL

    @property
    def deployments_info(self):
        """A list of DeploymentInfo entities that were loaded from the local source tree"""

        return self._deployments_info

    @property
    def datarobot_deployments(self):
        """A list of DataRobot deployment entities that were fetched from DataRobot."""

        return self._datarobot_deployments

    def scan_and_load_deployments_metadata(self):
        """Scan a load deployment definition yaml files from the load source stree."""

        logger.info("Scanning and loading DataRobot deployment files ...")
        for yaml_path, yaml_content in self._next_yaml_content_in_repo():
            if DeploymentSchema.is_multi_deployments_schema(yaml_content):
                transformed = DeploymentSchema.validate_and_transform_multi(yaml_content)
                for deployment_metadata in transformed:
                    deployment_info = DeploymentInfo(yaml_path, deployment_metadata)
                    self._add_new_deployment_info(deployment_info)
            elif DeploymentSchema.is_single_deployment_schema(yaml_content):
                transformed = DeploymentSchema.validate_and_transform_single(yaml_content)
                yaml_path = Path(yaml_path)
                deployment_info = DeploymentInfo(yaml_path, transformed)
                self._add_new_deployment_info(deployment_info)

    def _add_new_deployment_info(self, deployment_info):
        if deployment_info.user_provided_id in self._deployments_info:
            raise DeploymentMetadataAlreadyExists(
                f"Deployment {deployment_info.user_provided_id} already exists."
            )

        logger.info(
            "Adding new deployment metadata. User provided id: %s. "
            "Deployment metadata yaml path: %s.",
            deployment_info.user_provided_id,
            deployment_info.yaml_filepath,
        )
        self._deployments_info[deployment_info.user_provided_id] = deployment_info

    def fetch_deployments_from_datarobot(self):
        """Retrieve deployments entities from DataRobot."""

        logger.info("Fetching deployments from DataRobot ...")
        custom_inference_deployments = self._dr_client.fetch_deployments()
        for deployment in custom_inference_deployments:
            user_provided_id = deployment.get("userProvidedId")
            if user_provided_id:
                model_version = self._get_associated_model_version(user_provided_id, deployment)
                self.datarobot_deployments[user_provided_id] = DataRobotDeployment(
                    deployment, model_version
                )

    def _get_associated_model_version(self, user_provided_id, datarobot_deployment):
        model_id = datarobot_deployment["model"]["customModelImage"]["customModelId"]
        associated_datarobot_model = self._model_controller.datarobot_model_by_id(model_id)
        if not associated_datarobot_model:
            raise AssociatedModelNotFound(
                "Deployment is broken due to a missing associated datarobot model.\n"
                f"User provided deployment id: {user_provided_id}, model id: {model_id}."
            )
        if not associated_datarobot_model.latest_version:
            raise AssociatedModelVersionNotFound(
                "Deployment is broken due to a missing latest datarobot model version. "
                f"User provided deployment id: {user_provided_id}, model id: {model_id}."
            )

        datarobot_model_version_id = datarobot_deployment["model"]["customModelImage"][
            "customModelVersionId"
        ]
        if associated_datarobot_model.latest_version["id"] == datarobot_model_version_id:
            datarobot_model_version = associated_datarobot_model.latest_version
        else:
            datarobot_model_id = datarobot_deployment["model"]["customModelImage"]["customModelId"]
            datarobot_model_version = self._dr_client.fetch_custom_model_version(
                datarobot_model_id, datarobot_model_version_id
            )
        return datarobot_model_version

    def validate_deployments_integrity(self):
        """Validate deployments integrity with models."""

        logger.info("Validating deployments integrity ...")
        for user_provided_id, deployment_info in self.deployments_info.items():
            datarobot_deployment = self.datarobot_deployments.get(user_provided_id)
            if datarobot_deployment:
                model_version = datarobot_deployment.model_version
            else:
                # 1. Verify existing and valid local custom model that is associated with the
                #    deployment.
                model_info = self._model_controller.models_info.get(
                    deployment_info.user_provided_model_id
                )
                if not model_info:
                    raise AssociatedModelNotFound(
                        "Data integrity in local repository is broken. "
                        "There's no associated git model definition for given deployment. "
                        f"User provided deployment id: {user_provided_id}, "
                        f"User provided model id: {deployment_info.user_provided_model_id}."
                    )

                # 2. Validate that the associated model was already created in DataRobot
                custom_model = self._model_controller.datarobot_models.get(
                    deployment_info.user_provided_model_id
                )
                if not custom_model:
                    raise AssociatedModelNotFound(
                        "Unexpected missing DataRobot model. "
                        f"User provided deployment id: {user_provided_id}, "
                        f"User provided model id: {deployment_info.user_provided_model_id}."
                    )

                # 3. Validate at least one version
                if not custom_model.latest_version:
                    raise AssociatedModelVersionNotFound(
                        "Unexpected missing DataRobot model version. A custom model with at least "
                        "a single version must be created upfront. "
                        f"User provided deployment id: {user_provided_id}, "
                        f"User provided model id: {deployment_info.user_provided_model_id}, "
                        f"DataRobot model id: {custom_model.model['id']}."
                    )
                model_version = custom_model.latest_version

            # 4. Validate that the associated model's version SHA is an ancestor in the current tree
            git_main_branch_sha = model_version["gitModelVersion"]["mainBranchCommitSha"]
            if not self._repo.is_ancestor_of(git_main_branch_sha, "HEAD"):
                raise NoValidAncestor(
                    "The associated model's version git SHA is not an ancestor in the current "
                    f"branch. User provided deployment id: {user_provided_id}, "
                    f"Pinned sha: {git_main_branch_sha}."
                )

    def handle_deployment_changes_or_creation(self):
        """Apply changes to deployments in DataRobot."""

        for user_provided_id, deployment_info in self.deployments_info.items():
            datarobot_deployment = self.datarobot_deployments.get(user_provided_id)
            if not datarobot_deployment:
                self._create_deployment(deployment_info)
            else:
                # NOTE: settings changes should be applied before a replacement or a challenger
                # takes place, because we should first reflect the desired setting in DataRobot
                # and only then carry out these related actions.
                self._handle_deployment_changes(deployment_info, datarobot_deployment)

                active_datarobot_model_id = datarobot_deployment.model_version["customModelId"]
                desired_datarobot_model = self._model_controller.datarobot_models.get(
                    deployment_info.user_provided_model_id
                )
                if self._user_replaced_the_model_in_a_deployment(
                    desired_datarobot_model, active_datarobot_model_id
                ) or self._there_is_a_new_model_version(
                    desired_datarobot_model, datarobot_deployment
                ):
                    if deployment_info.is_challenger_enabled:
                        self._create_challenger_in_deployment(
                            desired_datarobot_model.latest_version,
                            datarobot_deployment,
                            deployment_info,
                        )
                    else:
                        self._replace_model_version_in_deployment(
                            desired_datarobot_model.latest_version, datarobot_deployment
                        )
                    self.stats.total_affected += 1

    def _create_deployment(self, deployment_info):
        logger.info(
            "Creating a deployment ... user_provided_id: %s.", deployment_info.user_provided_id
        )
        custom_model = self._model_controller.datarobot_models.get(
            deployment_info.user_provided_model_id
        )
        deployment = self._dr_client.create_deployment(custom_model.latest_version, deployment_info)
        logger.info(
            "A new deployment was created, git_id: %s, id: %s.",
            deployment_info.user_provided_id,
            deployment["id"],
        )

        self._handle_follow_up_deployment_settings(deployment_info, deployment)

        self.stats.total_created += 1
        self.stats.total_affected += 1

    def _handle_follow_up_deployment_settings(self, deployment_info, deployment):
        self._submit_actuals(deployment_info, deployment)

    def _submit_actuals(self, deployment_info, deployment):
        desired_association_id = deployment_info.get_settings_value(
            DeploymentSchema.ASSOCIATION_KEY, DeploymentSchema.ASSOCIATION_ACTUALS_ID_KEY
        )
        desired_dataset_id = deployment_info.get_settings_value(
            DeploymentSchema.ASSOCIATION_KEY, DeploymentSchema.ASSOCIATION_ACTUALS_DATASET_ID_KEY
        )
        if desired_association_id and desired_dataset_id:
            logger.info(
                "Submitting actuals for a deployment."
                "Git deployment ID: %s, actuals association ID: %s, actuals dataset ID: %s.",
                deployment_info.user_provided_id,
                desired_association_id,
                desired_dataset_id,
            )
            model_info = self._model_controller.models_info.get(
                deployment_info.user_provided_model_id
            )
            target_name = model_info.get_settings_value(ModelSchema.TARGET_NAME_KEY)
            self._dr_client.submit_deployment_actuals(
                target_name, desired_association_id, desired_dataset_id, deployment
            )

    @staticmethod
    def _user_replaced_the_model_in_a_deployment(
        desired_datarobot_model, active_datarobot_model_id
    ):
        return desired_datarobot_model.model["id"] != active_datarobot_model_id

    @staticmethod
    def _there_is_a_new_model_version(datarobot_model, datarobot_deployment):
        return datarobot_deployment.model_version["id"] != datarobot_model.latest_version["id"]

    def _replace_model_version_in_deployment(self, model_latest_version, datarobot_deployment):
        user_provided_id = datarobot_deployment.deployment["userProvidedId"]
        logger.info(
            "Replacing a model version in a deployment ... "
            "user_provided_id: %s, latest_version: %s.",
            user_provided_id,
            model_latest_version["id"],
        )
        deployment = self._dr_client.replace_model_deployment(
            model_latest_version, datarobot_deployment
        )
        logger.info(
            "The latest model version was successfully replaced in a deployment. "
            "user_provided_id: %s, deployment_id: %s.",
            user_provided_id,
            deployment["id"],
        )

    def _create_challenger_in_deployment(
        self, model_latest_version, datarobot_deployment, deployment_info
    ):
        user_provided_id = datarobot_deployment.deployment["userProvidedId"]
        logger.info(
            "Submitting a model challenger ... user provided deployment id: %s, "
            "model latest version id: %s.",
            user_provided_id,
            model_latest_version["id"],
        )
        challenger = self._dr_client.create_challenger(
            model_latest_version, datarobot_deployment, deployment_info
        )
        logger.info(
            "A challenger was successfully created and it is waiting for approval. "
            "user provided deployment id: %s, challenger id: %s.",
            user_provided_id,
            challenger["id"],
        )

    def _handle_deployment_changes(self, deployment_info, datarobot_deployment):
        desired_label = deployment_info.get_settings_value(DeploymentSchema.LABEL_KEY)
        if desired_label and desired_label != datarobot_deployment.deployment["label"]:
            datarobot_deployment_id = datarobot_deployment.deployment["id"]
            self._dr_client.update_deployment_label(datarobot_deployment_id, desired_label)

        self._handle_deployment_settings(deployment_info, datarobot_deployment)

    def _handle_deployment_settings(self, deployment_info, datarobot_deployment):
        datarobot_deployment_id = datarobot_deployment.deployment["id"]
        actual_deployment_settings = self._dr_client.fetch_deployment_settings(
            datarobot_deployment_id, deployment_info
        )
        self._dr_client.update_deployment_settings(
            datarobot_deployment_id, deployment_info, actual_deployment_settings
        )

        if self._dr_client.should_submit_new_actuals(deployment_info, actual_deployment_settings):
            self._submit_actuals(deployment_info, datarobot_deployment)

    def handle_deleted_deployments(self):
        """Delete deployments in DataRobot. Deletion takes place only within a push GitHub event."""

        if not GitHubEnv.is_push():
            logger.debug("Skip handling deployment deletion. It takes place only on push event.")
            return

        if not self.options.allow_deployment_deletion:
            logger.info("Skip handling deployment deletion because it is not enabled.")
            return

        logger.info("Deleting deployments (if any) ...")
        for user_provided_id, datarobot_deployment in self.datarobot_deployments.items():
            if user_provided_id not in self.deployments_info:
                deployment_id = datarobot_deployment.deployment["id"]
                try:
                    self._dr_client.delete_deployment_by_id(deployment_id)
                    self.stats.total_deleted += 1
                    self.stats.total_affected += 1
                    logger.info(
                        "A deployment was deleted with success. "
                        "user_provided_id: %s, deployment_id: %s.",
                        user_provided_id,
                        deployment_id,
                    )
                except DataRobotClientError as ex:
                    logger.error(str(ex))
