import logging
from pathlib import Path

from common.data_types import DataRobotDeployment
from common.exceptions import AssociatedModelNotFound
from common.exceptions import AssociatedModelVersionNotFound
from common.exceptions import DataRobotClientError
from common.exceptions import DeploymentMetadataAlreadyExists
from common.exceptions import NoValidAncestor
from custom_inference_model import CustomInferenceModelBase
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from schema_validator import SharedSchema

logger = logging.getLogger()


class DeploymentInfo:
    def __init__(self, yaml_path, deployment_metadata):
        self._yaml_path = yaml_path
        self._metadata = deployment_metadata

    @property
    def git_deployment_id(self):
        return self._metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]

    @property
    def git_model_id(self):
        return self._metadata[SharedSchema.MODEL_ID_KEY]

    @property
    def metadata(self):
        return self._metadata

    @property
    def yaml_filepath(self):
        return self._yaml_path

    @property
    def is_challenger_enabled(self):
        challenger_enabled = self.get_settings_value(DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY)
        return True if challenger_enabled is None else challenger_enabled

    def get_value(self, key, *args):
        return DeploymentSchema.get_value(self.metadata, key, *args)

    def set_value(self, key, *args, value):
        return DeploymentSchema.set_value(self.metadata, key, *args, value=value)

    def get_settings_value(self, key, *sub_keys):
        return self.get_value(DeploymentSchema.SETTINGS_SECTION_KEY, key, *sub_keys)

    def set_settings_value(self, key, *args, value):
        return self.set_value(DeploymentSchema.SETTINGS_SECTION_KEY, key, *args, value=value)


class CustomInferenceDeployment(CustomInferenceModelBase):
    def __init__(self, options):
        super().__init__(options)
        self._deployments_info = {}
        self._datarobot_deployments = {}

    @property
    def deployments_info(self):
        return self._deployments_info

    @property
    def datarobot_deployments(self):
        return self._datarobot_deployments

    def _label(self):
        return "deployments"

    def _run(self):
        """
        Executes the GitHub action logic to manage custom inference deployments
        """

        self._scan_and_load_deployments_metadata()
        self._fetch_models_from_datarobot()
        self._fetch_deployments_from_datarobot()
        self._validate_deployments_integrity()
        if self.event_name == "push":
            self._apply_datarobot_deployment_actions()

    def _scan_and_load_deployments_metadata(self):
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
        if deployment_info.git_deployment_id in self._deployments_info:
            raise DeploymentMetadataAlreadyExists(
                f"Deployment {deployment_info.git_deployment_id} already exists."
            )

        logger.info(
            f"Adding new deployment metadata. Git model id: {deployment_info.git_deployment_id}. "
            f"Deployment metadata yaml path: {deployment_info.yaml_filepath}."
        )
        self._deployments_info[deployment_info.git_deployment_id] = deployment_info

    def _fetch_deployments_from_datarobot(self):
        logger.info("Fetching deployments from DataRobot ...")
        custom_inference_deployments = self._dr_client.fetch_deployments()
        for deployment in custom_inference_deployments:
            git_deployment_id = deployment.get("gitDeploymentId")
            if git_deployment_id:
                model_version = self._get_associated_model_version(git_deployment_id, deployment)
                self.datarobot_deployments[git_deployment_id] = DataRobotDeployment(
                    deployment, model_version
                )

    def _get_associated_model_version(self, git_deployment_id, datarobot_deployment):
        model_id = datarobot_deployment["model"]["customModelImage"]["customModelId"]
        associated_datarobot_model = self.datarobot_model_by_id(model_id)
        if not associated_datarobot_model:
            raise AssociatedModelNotFound(
                "Deployment is broken due to a missing associated datarobot model.\n"
                f"Git deployment id: {git_deployment_id}, model id: {model_id}."
            )
        if not associated_datarobot_model.latest_version:
            raise AssociatedModelVersionNotFound(
                "Deployment is broken due to a missing latest datarobot model version. "
                f"Git deployment id: {git_deployment_id}, model id: {model_id}."
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

    def _validate_deployments_integrity(self):
        logger.info("Validating deployments integrity ...")
        for git_deployment_id, deployment_info in self.deployments_info.items():
            datarobot_deployment = self.datarobot_deployments.get(git_deployment_id)
            if datarobot_deployment:
                model_version = datarobot_deployment.model_version
            else:
                # 1. Verify existing and valid local custom model that is associated with the
                #    deployment.
                model_info = self.models_info.get(deployment_info.git_model_id)
                if not model_info:
                    raise AssociatedModelNotFound(
                        "Data integrity in local repository is broken. "
                        "There's no associated git model definition for given deployment. "
                        f"Git deployment id: {git_deployment_id}, "
                        f"git model id: {deployment_info.git_model_id}."
                    )

                # 2. Validate that the associated model was already created in DataRobot
                custom_model = self.datarobot_models.get(deployment_info.git_model_id)
                if not custom_model:
                    raise AssociatedModelNotFound(
                        "Unexpected missing DataRobot model. "
                        f"Git deployment id: {git_deployment_id}, "
                        f"git model id: {deployment_info.git_model_id}."
                    )

                # 3. Validate at least one version
                if not custom_model.latest_version:
                    raise AssociatedModelVersionNotFound(
                        "Unexpected missing DataRobot model version. A custom model with at least "
                        "a single version must be created upfront. "
                        f"Git deployment id: {git_deployment_id}, "
                        f"git model id: {deployment_info.git_model_id}, "
                        f"DataRobot model id: {custom_model.model['id']}."
                    )
                model_version = custom_model.latest_version

            # 4. Validate that the associated model's version SHA is an ancestor in the current tree
            git_main_branch_sha = model_version["gitModelVersion"]["mainBranchCommitSha"]
            if not self._repo.is_ancestor_of(git_main_branch_sha, "HEAD"):
                raise NoValidAncestor(
                    "The associated model's version git SHA is not an ancestor in the current "
                    "branch. "
                    f"Git deployment id: {git_deployment_id}, "
                    f"Pinned sha: {git_main_branch_sha}."
                )

    def _apply_datarobot_deployment_actions(self):
        logger.info("Applying DataRobot deployment actions ...")
        self._handle_deployment_changes_or_creation()
        self._handle_deleted_deployments()

    def _handle_deployment_changes_or_creation(self):
        for git_deployment_id, deployment_info in self.deployments_info.items():
            datarobot_deployment = self.datarobot_deployments.get(git_deployment_id)
            if not datarobot_deployment:
                self._create_deployment(deployment_info)
            else:
                datarobot_model = self.datarobot_models.get(deployment_info.git_model_id)
                if self._there_is_a_new_model_version(datarobot_model, datarobot_deployment):
                    if deployment_info.is_challenger_enabled:
                        self._create_challenger_in_deployment(
                            datarobot_model.latest_version, datarobot_deployment, deployment_info
                        )
                    else:
                        self._replace_model_version_in_deployment(
                            datarobot_model.latest_version, datarobot_deployment
                        )

                self._handle_deployment_changes(deployment_info, datarobot_deployment)

    def _create_deployment(self, deployment_info):
        logger.info(
            f"Creating a deployment ... git_deployment_id: {deployment_info.git_deployment_id}"
        )
        custom_model = self.datarobot_models.get(deployment_info.git_model_id)
        deployment = self._dr_client.create_deployment(custom_model.latest_version, deployment_info)
        logger.info(
            "A new deployment was created, "
            f"git_id: {deployment_info.git_deployment_id}, id: {deployment['id']}"
        )

        self._handle_follow_up_deployment_settings(deployment_info, deployment)

        self._total_created += 1
        self._total_affected += 1

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
                f"Git deployment ID: {deployment_info.git_deployment_id}, "
                f"actuals association ID: {desired_association_id}, "
                f"actuals dataset ID: {desired_dataset_id}, "
            )
            model_info = self.models_info.get(deployment_info.git_model_id)
            target_name = model_info.get_settings_value(ModelSchema.TARGET_NAME_KEY)
            self._dr_client.submit_deployment_actuals(
                target_name, desired_association_id, desired_dataset_id, deployment
            )

    @staticmethod
    def _there_is_a_new_model_version(datarobot_model, datarobot_deployment):
        return datarobot_deployment.model_version["id"] != datarobot_model.latest_version["id"]

    def _replace_model_version_in_deployment(self, model_latest_version, datarobot_deployment):
        git_deployment_id = datarobot_deployment.deployment["gitDeploymentId"]
        logger.info(
            f"Replacing a model version in a deployment ... "
            f"git_deployment_id: {git_deployment_id}, "
            f"latest_version: {model_latest_version['id']}."
        )
        deployment = self._dr_client.replace_model_deployment(
            model_latest_version, datarobot_deployment
        )
        self._total_affected += 1
        logger.info(
            f"The latest model version was successfully replaced in a deployment. "
            f"git_deployment_id: {git_deployment_id}."
            f"deployment_id: {deployment['id']}."
        )

    def _create_challenger_in_deployment(
        self, model_latest_version, datarobot_deployment, deployment_info
    ):
        git_deployment_id = datarobot_deployment.deployment["gitDeploymentId"]
        logger.info(
            f"Submitting a model challenger ... "
            f"git_deployment_id: {git_deployment_id}, "
            f"latest_version: {model_latest_version['id']}."
        )
        challenger = self._dr_client.create_challenger(
            model_latest_version, datarobot_deployment, deployment_info
        )
        logger.info(
            f"A challenger was successfully created and it is waiting for approval. "
            f"git_deployment_id: {git_deployment_id}."
            f"challenger_id: {challenger['id']}."
        )

    def _handle_deployment_changes(self, deployment_info, datarobot_deployment):
        desired_label = deployment_info.get_settings_value(DeploymentSchema.LABEL_KEY)
        if desired_label != datarobot_deployment.deployment["label"]:
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

        if self._should_submit_new_actuals(deployment_info, actual_deployment_settings):
            self._submit_actuals(deployment_info, datarobot_deployment)

    @staticmethod
    def _should_submit_new_actuals(deployment_info, actual_settings):
        """
        New actuals are assumed to be submitted only if the association ID was changed.

        Parameters
        ----------
        deployment_info : DeploymentInfo
            The deployment info.
        actual_settings : dict
            The actual deployment settings from DataRobot.

        Returns
        -------
        bool,
            Whether the actuals should be submitted or not
        """

        desired_association_section = deployment_info.get_settings_value(
            DeploymentSchema.ASSOCIATION_KEY
        )
        if desired_association_section is not None:
            desired_association_pred_id = deployment_info.get_settings_value(
                DeploymentSchema.ASSOCIATION_KEY, DeploymentSchema.ASSOCIATION_PRED_ID_KEY
            )
            if desired_association_pred_id is not None:
                actuals_cols = (
                    actual_settings["associationId"]["columnNames"] if actual_settings else None
                )
                return [desired_association_pred_id] != actuals_cols
        return False

    def _handle_deleted_deployments(self):
        if not self.is_push:
            logger.debug("Skip handling deployment deletion. It takes place only on push event.")
            return

        if not self.options.allow_deployment_deletion:
            logger.info("Skip handling deployment deletion because it is not enabled.")
            return

        logger.info("Deleting deployments (if any) ...")
        for git_deployment_id, datarobot_deployment in self.datarobot_deployments.items():
            if git_deployment_id not in self.deployments_info:
                deployment_id = datarobot_deployment.deployment["id"]
                # TODO: skip deletion of 'dirty' deployments. Only show a warning.
                try:
                    self._dr_client.delete_deployment_by_id(deployment_id)
                    self._total_deleted += 1
                    self._total_affected += 1
                    logger.info(
                        "A deployment was deleted with success. "
                        f"git_deployment_id: {git_deployment_id}, deployment_id: {deployment_id}."
                    )
                except DataRobotClientError as ex:
                    logger.error(str(ex))
