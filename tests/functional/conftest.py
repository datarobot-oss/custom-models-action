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

from common.convertors import MemoryConvertor
from common.exceptions import DataRobotClientError
from common.exceptions import IllegalModelDeletion
from dr_client import DrClient
from main import main
from schema_validator import ModelSchema
from schema_validator import SharedSchema


def webserver_accessible():
    webserver = os.environ.get("DATAROBOT_WEBSERVER")
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    if webserver and api_token:
        return DrClient(webserver, api_token, verify_cert=False).is_accessible()
    return False


def cleanup_models(dr_client, repo_root_path):
    for model_yaml_file in repo_root_path.rglob("**/model.yaml"):
        with open(model_yaml_file) as f:
            model_metadata = yaml.safe_load(f)

        try:
            dr_client.delete_custom_model_by_git_model_id(model_metadata[ModelSchema.MODEL_ID_KEY])
        except (IllegalModelDeletion, DataRobotClientError):
            pass


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
    dst_models_dir = repo_root_path / models_src_root_dir.name
    shutil.copytree(models_src_root_dir, dst_models_dir)

    # 2. Rename first model to indicate '1'
    model_filepath_1 = next(dst_models_dir.glob("*"))
    first_model_filepath = str(model_filepath_1) + "_1"
    model_filepath_1 = model_filepath_1.rename(first_model_filepath)

    # 3. Duplicate the first model to simulate more than one model
    second_model_filepath = first_model_filepath.replace("_1", "_2")
    shutil.copytree(first_model_filepath, second_model_filepath)

    second_model_metadata_yaml_filepath = next(Path(second_model_filepath).rglob("**/model.yaml"))
    with open(second_model_metadata_yaml_filepath, "r") as f:
        second_model_metadata = yaml.safe_load(f)

    # 4. Change second model ID
    second_model_id = second_model_metadata[ModelSchema.MODEL_ID_KEY]
    second_model_id = second_model_id.replace("1", "2")
    ModelSchema.set_value(second_model_metadata, ModelSchema.MODEL_ID_KEY, second_model_id)

    # 5. Change second model name
    second_model_name = ModelSchema.get_value(
        second_model_metadata, ModelSchema.SETTINGS_SECTION_KEY, ModelSchema.NAME_KEY
    )
    second_model_name = second_model_name.replace("1", "2")
    ModelSchema.set_value(
        second_model_metadata,
        ModelSchema.SETTINGS_SECTION_KEY,
        ModelSchema.NAME_KEY,
        second_model_name,
    )

    # 6. Save second model metadata
    with open(second_model_metadata_yaml_filepath, "w") as f:
        yaml.safe_dump(second_model_metadata, f)

    # 7. Copy deployments
    deployments_src_root_dir = Path(__file__).parent / ".." / "deployments"
    dst_deployments_dir = repo_root_path / deployments_src_root_dir.name
    shutil.copytree(deployments_src_root_dir, dst_deployments_dir)

    # 8. Add files to repo
    os.chdir(repo_root_path)
    git_repo.git.add("--all")
    git_repo.git.commit("-m", "Initial commit", "--no-verify")


@pytest.fixture
def set_model_dataset_for_testing(dr_client, model_metadata, model_metadata_yaml_file):
    if ModelSchema.TEST_KEY in model_metadata:
        test_dataset_filepath = (
            Path(__file__).parent / ".." / "datasets" / "juniors_3_year_stats_regression_small.csv"
        )
        dataset_id = None
        try:
            dataset_id = dr_client.upload_dataset(test_dataset_filepath)

            with set_persistent_schema_variable(
                model_metadata_yaml_file,
                model_metadata,
                dataset_id,
                ModelSchema.TEST_KEY,
                ModelSchema.TEST_DATA_KEY,
            ):
                yield dataset_id
        finally:
            if dataset_id:
                dr_client.delete_dataset(dataset_id)


@contextlib.contextmanager
def set_persistent_schema_variable(yaml_filepath, metadata, new_value, *args):
    try:
        origin_value = SharedSchema.get_value(metadata, *args)
        SharedSchema.set_value(metadata, *args, new_value)
        with open(yaml_filepath, "w") as f:
            yaml.safe_dump(metadata, f)

        yield

    finally:
        args = args + (origin_value,)
        SharedSchema.set_value(metadata, *args)
        with open(yaml_filepath, "w") as f:
            yaml.safe_dump(metadata, f)


@pytest.fixture
def skip_model_testing(model_metadata, model_metadata_yaml_file):
    origin_test_section = model_metadata.get(ModelSchema.TEST_KEY)
    model_metadata.pop(ModelSchema.TEST_KEY, None)
    with open(model_metadata_yaml_file, "w") as f:
        yaml.safe_dump(model_metadata, f)

    yield

    model_metadata[ModelSchema.TEST_KEY] = origin_test_section
    with open(model_metadata_yaml_file, "w") as f:
        yaml.safe_dump(model_metadata, f)


# NOTE: it was rather better to use the pytest.mark.usefixture for 'build_repo_for_testing'
# but, apparently it cannot be used with fixtures.
@pytest.fixture
def model_metadata_yaml_file(build_repo_for_testing, repo_root_path, git_repo):
    model_yaml_file = next(repo_root_path.rglob("*_1/model.yaml"))
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


def increase_model_memory_by_1mb(model_yaml_file):
    with open(model_yaml_file) as f:
        yaml_content = yaml.safe_load(f)
        memory = ModelSchema.get_value(
            yaml_content, ModelSchema.VERSION_KEY, ModelSchema.MEMORY_KEY
        )
        memory = memory if memory else "256Mi"
        num_part, unit = MemoryConvertor._extract_unit_fields(memory)
        new_memory = f"{num_part+1}{unit}"
        yaml_content[ModelSchema.VERSION_KEY][ModelSchema.MEMORY_KEY] = new_memory

    with open(model_yaml_file, "w") as f:
        yaml.safe_dump(yaml_content, f)

    return new_memory


def run_github_action(
    repo_root_path,
    git_repo,
    main_branch_name,
    main_branch_head_sha,
    event_name,
    is_deploy,
    allow_deployment_deletion=True,
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

        if allow_deployment_deletion:
            args.append("--allow-deployment-deletion")

        main(args)
