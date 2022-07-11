import logging
from pathlib import Path

from common.exceptions import AssociatedModelNotFound
from common.exceptions import AssociatedModelVersionNotFound
from common.exceptions import DeploymentMetadataAlreadyExists
from common.exceptions import NoValidAncestor
from common.exceptions import UnexpectedNumOfModelVersions
from custom_inference_model import CustomInferenceModelBase
from schema_validator import DeploymentSchema

logger = logging.getLogger()


class DeploymentInfo:
    def __init__(self, yaml_path, deployment_metadata):
        self._yaml_path = yaml_path
        self._metadata = deployment_metadata

    @property
    def git_deployment_id(self):
        return self._metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]

    @property
    def metadata(self):
        return self._metadata

    @property
    def yaml_filepath(self):
        return self._yaml_path

    @property
    def pinned_model_sha(self):
        model_sha = DeploymentSchema.get_value(self.metadata, DeploymentSchema.MODEL_SHA_KEY)
        if model_sha == DeploymentSchema.LATEST_SHA_VALUE:
            return None
        return model_sha


class CustomInferenceDeployment(CustomInferenceModelBase):
    def __init__(self, options):
        super().__init__(options)
        self._model_id_to_model = {}
        self._deployments_info = []
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
        # if self.event_name == "push":
        #     self._create_or_change_deployments()
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

    def _validate_deployments_integrity(self):
        logger.info("Validating deployments integrity ...")
        # 1. For every local deployment
        for deployment_info in self.deployments_info:
            # 2. Get corresponding DataRobot deployment
            datarobot_deployment = self.datarobot_deployments.get(deployment_info.git_deployment_id)
            if not datarobot_deployment:
                # Will be created in the next stage
                continue

            # 3. Validate associated DataRobot model
            model_id = datarobot_deployment["model"]["customModelImage"]["customModelId"]
            associated_datarobot_model = self.datarobot_model_by_id(model_id)
            if not associated_datarobot_model:
                raise AssociatedModelNotFound(
                    "Deployment is broken due to a missing associated datarobot model.\n"
                    f"Git deployment ID: {deployment_info.git_deployment_id}, "
                    f"Missing datarobot model ID: {model_id}."
                )
            if not associated_datarobot_model.latest_version:
                raise AssociatedModelVersionNotFound(
                    "Deployment is broken due to a missing associated datarobot model version."
                    f"Git deployment ID: {deployment_info.git_deployment_id}, "
                    f"DataRobot model ID: {model_id}, missing DataRobot model ID: {model_id}."
                )

            if deployment_info.pinned_model_sha:
                git_main_branch_sha = deployment_info.pinned_model_sha

                # 4. Validate that there's one and only one custom model version with the given
                #      model SHA
                cm_versions = self._dr_client.fetch_custom_model_versions(
                    associated_datarobot_model.model["id"],
                    json={"mainBranchCommitSha": deployment_info.pinned_model_sha},
                )
                if not cm_versions:
                    raise AssociatedModelVersionNotFound(
                        "Model version was not found in DataRobot for the given pinned model SHA."
                        f"Git deployment ID: {deployment_info.git_deployment_id}, "
                        f"pinned SHA: {deployment_info.pinned_model_sha}."
                    )
                if len(cm_versions) != 1:
                    raise UnexpectedNumOfModelVersions(
                        "Unexpected number of model versions for a given git main branch SHA. "
                        f"Git deployment ID: {deployment_info.git_deployment_id}, "
                        f"pinned SHA: {deployment_info.pinned_model_sha}, "
                        f"Num of model versions: {len(cm_versions)}."
                    )
            else:
                git_main_branch_sha = associated_datarobot_model.latest_version["gitModelVersion"][
                    "gitMainBranchSha"
                ]

            # 5. Validate that the associated model's version SHA is an ancestor in the current tree
            if not self._repo.is_ancestor_of(git_main_branch_sha, "HEAD"):
                raise NoValidAncestor(
                    "The associated model's version git SHA is not an ancestor in the current "
                    "branch. "
                    f"Git deployment ID: {deployment_info.git_deployment_id}, "
                    f"Pinned sha: {git_main_branch_sha}."
                )
