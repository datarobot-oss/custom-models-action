import contextlib
import os
from pathlib import Path

import pytest
import yaml
from bson import ObjectId

from common.exceptions import DataRobotClientError
from common.exceptions import IllegalModelDeletion
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from tests.functional.conftest import cleanup_models
from tests.functional.conftest import increase_model_memory_by_1mb
from tests.functional.conftest import run_github_action
from tests.functional.conftest import set_persistent_schema_variable
from tests.functional.conftest import webserver_accessible


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
def cleanup(dr_client, repo_root_path, deployment_metadata):
    yield

    try:
        dr_client.delete_deployment_by_user_id(
            deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        )
    except (IllegalModelDeletion, DataRobotClientError):
        pass

    # NOTE: we have more than one model in the tree
    cleanup_models(dr_client, repo_root_path)


@pytest.mark.skipif(not webserver_accessible(), reason="DataRobot webserver is not accessible.")
@pytest.mark.usefixtures("build_repo_for_testing")
class TestDeploymentGitHubActions:
    @contextlib.contextmanager
    def _upload_actuals_dataset(
        self, event_name, dr_client, deployment_metadata, deployment_metadata_yaml_file
    ):
        if event_name == "push":
            association_id = DeploymentSchema.get_value(
                deployment_metadata,
                DeploymentSchema.SETTINGS_SECTION_KEY,
                DeploymentSchema.ASSOCIATION_ID_KEY,
            )
            if association_id:
                actuals_filepath = (
                    Path(__file__).parent
                    / ".."
                    / "datasets"
                    / "juniors_3_year_stats_regression_small_actuals.csv"
                )
                dataset_id = None
                try:
                    dataset_id = dr_client.upload_dataset(actuals_filepath)
                    with set_persistent_schema_variable(
                        deployment_metadata_yaml_file,
                        deployment_metadata,
                        dataset_id,
                        DeploymentSchema.SETTINGS_SECTION_KEY,
                        DeploymentSchema.ACTUALS_DATASET_ID_KEY,
                    ):
                        yield dataset_id
                finally:
                    if dataset_id:
                        dr_client.delete_dataset(dataset_id)
        else:
            yield None

    @pytest.mark.parametrize("event_name", ["push", "pull_request"])
    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_deployment_create(
        self,
        dr_client,
        repo_root_path,
        git_repo,
        model_metadata,
        model_metadata_yaml_file,
        deployment_metadata,
        deployment_metadata_yaml_file,
        main_branch_name,
        event_name,
    ):
        # 1. Create a model just as a preliminary requirement (use GitHub action)
        head_commit_sha = git_repo.head.commit.hexsha
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=False
        )

        # 2. Upload actuals dataset and set the deployment metadata with the dataset ID
        with self._upload_actuals_dataset(
            event_name, dr_client, deployment_metadata, deployment_metadata_yaml_file
        ):
            # 3. Run a deployment github action
            run_github_action(
                repo_root_path,
                git_repo,
                main_branch_name,
                head_commit_sha,
                event_name,
                is_deploy=True,
            )

        deployments = dr_client.fetch_deployments()
        local_git_deployment_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        if event_name == "push":
            assert any(d.get("gitDeploymentId") == local_git_deployment_id for d in deployments)
        elif event_name == "pull_request":
            assert all(d.get("gitDeploymentId") != local_git_deployment_id for d in deployments)
        else:
            assert False, f"Unsupported GitHub event name: {event_name}"

    @pytest.mark.parametrize("event_name", ["push", "pull_request"])
    @pytest.mark.usefixtures("cleanup", "set_model_dataset_for_testing")
    def test_e2e_deployment_model_replacement(
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
        # Disable challengers
        self._enable_challenger(deployment_metadata, deployment_metadata_yaml_file, False)

        (
            _,
            latest_deployment_model_version_id,
            latest_model_version,
        ) = self._deploy_a_model_than_run_github_action_to_replace_or_challenge(
            dr_client,
            git_repo,
            repo_root_path,
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

    @staticmethod
    def _deploy_a_model_than_run_github_action_to_replace_or_challenge(
        dr_client,
        git_repo,
        repo_root_path,
        main_branch_name,
        deployment_metadata,
        model_metadata_yaml_file,
        event_name,
    ):

        # 1. Create a model just as a basic requirement (use GitHub action)
        head_commit_sha = git_repo.head.commit.hexsha
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=False
        )

        # 2. Create a deployment
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=True
        )
        local_git_deployment_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        deployments = dr_client.fetch_deployments()
        assert any(d.get("gitDeploymentId") == local_git_deployment_id for d in deployments)

        # 3. Make a local change to the model and commit
        new_memory = increase_model_memory_by_1mb(model_metadata_yaml_file)
        os.chdir(repo_root_path)
        git_repo.git.commit("-a", "-m", f"Increase memory to {new_memory}")

        # 4. Run GitHub action to create a new model version in DataRobot
        head_commit_sha = git_repo.head.commit.hexsha
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=False
        )

        # 5. Run GitHub action to replace the latest model in a deployment
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, event_name, is_deploy=True
        )

        deployments = dr_client.fetch_deployments()
        the_deployment = next(
            d for d in deployments if d.get("gitDeploymentId") == local_git_deployment_id
        )

        latest_deployment_model_version_id = the_deployment["model"]["customModelImage"][
            "customModelVersionId"
        ]
        local_git_model_id = deployment_metadata[DeploymentSchema.MODEL_ID_KEY]
        latest_model_version = dr_client.fetch_custom_model_latest_version_by_git_model_id(
            local_git_model_id
        )
        return the_deployment, latest_deployment_model_version_id, latest_model_version

    @staticmethod
    def _enable_challenger(deployment_metadata, deployment_metadata_yaml_file, enabled=True):
        settings = deployment_metadata.get(DeploymentSchema.SETTINGS_SECTION_KEY, {})
        settings[DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY] = enabled
        deployment_metadata[DeploymentSchema.SETTINGS_SECTION_KEY] = settings
        with open(deployment_metadata_yaml_file, "w") as f:
            yaml.safe_dump(deployment_metadata, f)

    @pytest.mark.usefixtures("cleanup", "skip_model_testing")
    def test_e2e_deployment_delete(
        self,
        dr_client,
        repo_root_path,
        git_repo,
        model_metadata_yaml_file,
        deployment_metadata,
        deployment_metadata_yaml_file,
        main_branch_name,
    ):
        # 1. Create a model just as a basic requirement (use GitHub action)
        head_commit_sha = git_repo.head.commit.hexsha
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=False
        )

        # 2. Run a deployment GitHub action to create a deployment
        run_github_action(
            repo_root_path, git_repo, main_branch_name, head_commit_sha, "push", is_deploy=True
        )
        deployments = dr_client.fetch_deployments()
        local_git_deployment_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        assert any(d.get("gitDeploymentId") == local_git_deployment_id for d in deployments)

        # 3. Delete a deployment local definition yaml file
        os.remove(deployment_metadata_yaml_file)
        os.chdir(repo_root_path)
        git_repo.git.commit("-a", "-m", f"Delete the deployment definition file")

        # 4. Run a deployment GitHub action but disallow deployment deletion
        run_github_action(
            repo_root_path,
            git_repo,
            main_branch_name,
            head_commit_sha,
            "push",
            is_deploy=True,
            allow_deployment_deletion=False,
        )
        deployments = dr_client.fetch_deployments()
        local_git_deployment_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        assert any(d.get("gitDeploymentId") == local_git_deployment_id for d in deployments)

        # 5. Run a deployment GitHub action for pull request with allowed deployment deletion
        run_github_action(
            repo_root_path,
            git_repo,
            main_branch_name,
            head_commit_sha,
            "pull_request",
            is_deploy=True,
            allow_deployment_deletion=True,
        )
        deployments = dr_client.fetch_deployments()
        local_git_deployment_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        assert any(d.get("gitDeploymentId") == local_git_deployment_id for d in deployments)

        # 6. Run a deployment GitHub action for push with allowed deployment deletion
        run_github_action(
            repo_root_path,
            git_repo,
            main_branch_name,
            head_commit_sha,
            "push",
            is_deploy=True,
            allow_deployment_deletion=True,
        )
        deployments = dr_client.fetch_deployments()
        local_git_deployment_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        assert all(d.get("gitDeploymentId") != local_git_deployment_id for d in deployments)

    @pytest.mark.parametrize("event_name", ["push", "pull_request"])
    @pytest.mark.usefixtures("cleanup", "set_model_dataset_for_testing")
    def test_e2e_deployment_model_challengers(
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
        # Enable challengers (although it is the default)
        self._enable_challenger(deployment_metadata, deployment_metadata_yaml_file, True)

        (
            the_deployment,
            latest_deployment_model_version_id,
            latest_model_version,
        ) = self._deploy_a_model_than_run_github_action_to_replace_or_challenge(
            dr_client,
            git_repo,
            repo_root_path,
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
