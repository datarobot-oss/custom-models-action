import pytest
import yaml
from bson import ObjectId

from common.exceptions import DataRobotClientError
from common.exceptions import IllegalModelDeletion
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from tests.functional.conftest import run_github_action


@pytest.fixture
@pytest.mark.usefixtures("build_repo_for_testing")
def deployment_metadata_yaml_file(repo_root_path, git_repo, model_metadata):
    deployment_yaml_file = next(repo_root_path.rglob("**/deployment.yaml"))
    with open(deployment_yaml_file) as f:
        yaml_content = yaml.safe_load(f)
        yaml_content[DeploymentSchema.DEPLOYMENT_ID_KEY] = f"deployment-id-{str(ObjectId())}"
        yaml_content[DeploymentSchema.MODEL_ID_KEY] = model_metadata[ModelSchema.MODEL_ID_KEY]

    with open(deployment_yaml_file, "w") as f:
        yaml.safe_dump(yaml_content, f)

    git_repo.git.add(deployment_yaml_file)
    git_repo.git.commit("--amend", "--no-edit")

    return deployment_yaml_file


@pytest.fixture
def deployment_metadata(deployment_metadata_yaml_file):
    with open(deployment_metadata_yaml_file) as f:
        return yaml.safe_load(f)


@pytest.fixture
def cleanup(dr_client, model_metadata, deployment_metadata):
    yield

    try:
        dr_client.delete_deployment_by_user_id(
            deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        )
    except (IllegalModelDeletion, DataRobotClientError):
        pass

    try:
        dr_client.delete_custom_model_by_git_model_id(model_metadata[ModelSchema.MODEL_ID_KEY])
    except (IllegalModelDeletion, DataRobotClientError):
        pass


@pytest.mark.parametrize("event_name", ["push", "pull_request"])
@pytest.mark.usefixtures("build_repo_for_testing", "upload_dataset_for_testing")
class TestDeploymentGitHubActions:
    @pytest.mark.usefixtures("cleanup")
    def test_e2e_deployment_create(
        self,
        dr_client,
        repo_root_path,
        git_repo,
        model_metadata_yaml_file,
        deployment_metadata,
        deployment_metadata_yaml_file,
        main_branch_name,
        event_name,
    ):
        # 1. Create model just as a preparation
        head_commit_sha = git_repo.head.commit.hexsha
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=False
        )

        # 2. Run a deployment github action
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, event_name, is_deploy=True
        )
        deployments = dr_client.fetch_deployments()
        local_git_deployment_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        if event_name == "push":
            assert any(d.get("gitDeploymentId") == local_git_deployment_id for d in deployments)
        elif event_name == "pull_request":
            assert all(d.get("gitDeploymentId") != local_git_deployment_id for d in deployments)
        else:
            assert False, f"Unsupported GitHub event name: {event_name}"
