import logging
from pathlib import Path

from common.data_types import DataRobotDeployment
from common.exceptions import AssociatedModelNotFound
from common.exceptions import AssociatedModelVersionNotFound
from common.exceptions import DeploymentMetadataAlreadyExists
from common.exceptions import NoValidAncestor
from custom_inference_model import CustomInferenceModelBase
from schema_validator import DeploymentSchema
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


class CustomInferenceDeployment(CustomInferenceModelBase):
    def __init__(self, options):
        super().__init__(options)
        self._model_id_to_model = {}
        self._deployments_info = {}
        self._datarobot_deployments = {}

    @property
    def deployments_info(self):
        return self._deployments_info

    @property
    def datarobot_deployments(self):
        return self._datarobot_deployments

    def _run(self):
        """
        Executes the GitHub action logic to manage custom inference deployments
        """

        self._scan_and_load_deployments_metadata()
        self._fetch_models_from_datarobot()
        self._fetch_deployments_from_datarobot()
        self._validate_deployments_integrity()
        if self.event_name == "push":
            self._create_or_change_deployments()
        #     self._deplete_deployments()

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

    def _create_or_change_deployments(self):
        for git_deployment_id, deployment_info in self.deployments_info.items():
            datarobot_deployment = self.datarobot_deployments.get(git_deployment_id)
            if not datarobot_deployment:
                logger.info("Creating a deployment ...")
                self._create_deployment(deployment_info)
            else:
                logger.info("[TODO] Updating a deployment ...")
                # 1. Check if a model version needs to be replaced
                # 2. Check if settings needs to be updated

    def _create_deployment(self, deployment_info):
        custom_model = self.datarobot_models.get(deployment_info.git_model_id)
        deployment = self._dr_client.create_deployment(custom_model.latest_version, deployment_info)
        logger.info(f"A new deployment was created, id: {deployment['id']}")
