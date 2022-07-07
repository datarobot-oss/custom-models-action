import logging
from glob import glob
from pathlib import Path

from common.exceptions import DeploymentMetadataAlreadyExists
from custom_inference_model import CustomInferenceModelBase
from schema_validator import DeploymentSchema

logger = logging.getLogger()


class DeploymentInfo:
    def __init__(self, yaml_path, deployment_metadata):
        self._yaml_path = yaml_path
        self._deployment_metadata = deployment_metadata

    @property
    def git_deployment_id(self):
        return self._deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]

    @property
    def yaml_filepath(self):
        return self._yaml_path


class CustomInferenceDeployment(CustomInferenceModelBase):
    def __init__(self, options):
        super().__init__(options)
        self._deployments_info = []
        self._datarobot_deployments = {}

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
        try:
            already_exists = next(
                d
                for d in self._deployments_info
                if d.git_deployment_id == deployment_info.git_deployment_id
            )
            raise DeploymentMetadataAlreadyExists(
                f"Deployment {deployment_info.git_deployment_id} already exists. "
                f"New deployment yaml path: {deployment_info.yaml_filepath}. "
                f"Existing model yaml path: {already_exists.yaml_filepath}."
            )
        except StopIteration:
            pass

        logger.info(
            f"Adding new deployment metadata. Git model ID: {deployment_info.git_deployment_id}. "
            f"Deployment metadata yaml path: {deployment_info.yaml_filepath}."
        )
        self._deployments_info.append(deployment_info)

    def _fetch_deployments_from_datarobot(self):
        logger.info("Fetching deployments from DataRobot ...")
        custom_inference_deployments = self._dr_client.fetch_deployments()
        for deployment in custom_inference_deployments:
            git_deployment_id = deployment.get("gitDeploymentId")
            if git_deployment_id:
                self.datarobot_deployments[git_deployment_id] = deployment
