import contextlib
import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml
from bson import ObjectId
from git import Repo

from dr_client import DrClient
from main import main
from schema_validator import ModelSchema


def webserver_accessible():
    webserver = os.environ.get("DATAROBOT_WEBSERVER")
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    if webserver and api_token:
        return DrClient(webserver, api_token, verify_cert=False).is_accessible()
    return False


@contextlib.contextmanager
def github_env_set(env_key, env_value):
    does_exist = env_key in os.environ
    if does_exist:
        old_value = os.environ[env_key]
    os.environ[env_key] = env_value
    yield
    if does_exist:
        os.environ[env_key] = old_value


@pytest.fixture
def repo_root_path():
    with TemporaryDirectory() as repo_tree:
        path = Path(repo_tree)
        yield path


@pytest.fixture
def git_repo(repo_root_path):
    repo = Repo.init(repo_root_path)
    repo.config_writer().set_value("user", "name", "functional-test-user").release()
    repo.config_writer().set_value("user", "email", "functional-test@company.com").release()
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    type(repo.git).GIT_PYTHON_TRACE = "full"
    return repo


@pytest.fixture
def build_repo_for_testing(repo_root_path, git_repo):
    # 1. Copy models from source tree
    models_src_root_dir = Path(__file__).parent / ".." / "models"
    shutil.copytree(models_src_root_dir, repo_root_path / models_src_root_dir.name)

    # 2. Add files to repo
    os.chdir(repo_root_path)
    git_repo.git.add("--all")
    git_repo.git.commit("-m", "Initial commit", "--no-verify")


@pytest.fixture
def upload_dataset_for_testing(dr_client, model_metadata):
    if ModelSchema.TEST_KEY in model_metadata:
        test_dataset_filepath = (
            Path(__file__).parent / ".." / "datasets" / "juniors_3_year_stats_regression_small.csv"
        )
        dataset_id = dr_client.upload_dataset(test_dataset_filepath)
        model_metadata[ModelSchema.TEST_KEY][ModelSchema.TEST_DATA_KEY] = dataset_id

        yield dataset_id
        dr_client.delete_dataset(dataset_id)


# NOTE: it was rather better to use the pytest.mark.usefixture for 'build_repo_for_testing'
# but, apparently it cannot be used with fixtures.
@pytest.fixture
def model_metadata_yaml_file(build_repo_for_testing, repo_root_path, git_repo):
    model_yaml_file = next(repo_root_path.rglob("**/model.yaml"))
    with open(model_yaml_file) as f:
        yaml_content = yaml.safe_load(f)
        yaml_content[ModelSchema.MODEL_ID_KEY] = f"my-awesome-model-{str(ObjectId())}"

    with open(model_yaml_file, "w") as f:
        yaml.safe_dump(yaml_content, f)

    git_repo.git.add(model_yaml_file)
    git_repo.git.commit("--amend", "--no-edit")

    return model_yaml_file


@pytest.fixture
def model_metadata(model_metadata_yaml_file):
    with open(model_metadata_yaml_file) as f:
        return yaml.safe_load(f)


@pytest.fixture
def main_branch_name():
    return "master"


@pytest.fixture
def dr_client():
    webserver = os.environ.get("DATAROBOT_WEBSERVER")
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    return DrClient(webserver, api_token, verify_cert=False)


def run_github_action(
    repo_root_path, git_repo, main_branch_name, main_branch_head_sha, event_name, is_deploy
):
    with github_env_set("GITHUB_EVENT_NAME", event_name), github_env_set(
        "GITHUB_SHA", git_repo.commit(main_branch_head_sha).hexsha
    ), github_env_set("GITHUB_BASE_REF", main_branch_name):
        args = [
            "--webserver",
            os.environ.get("DATAROBOT_WEBSERVER"),
            "--skip-cert-verification",
            "--api-token",
            os.environ.get("DATAROBOT_API_TOKEN"),
            "--branch",
            main_branch_name,
            "--root-dir",
            str(repo_root_path),
            "--allow-model-deletion",
        ]
        if is_deploy:
            args.append("--deploy")

        main(args)
