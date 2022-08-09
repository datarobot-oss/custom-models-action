import contextlib
import copy
import logging
import os
import shutil
from pathlib import Path
import random
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
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from schema_validator import SharedSchema


def webserver_accessible():
    webserver = os.environ.get("DATAROBOT_WEBSERVER")
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    if webserver and api_token:
        return DrClient(webserver, api_token, verify_cert=False).is_accessible()
    return False


def cleanup_models(dr_client, repo_root_path):
    custom_models = dr_client.fetch_custom_models()
    if custom_models:
        for model_yaml_file in repo_root_path.rglob("**/model.yaml"):
            with open(model_yaml_file) as f:
                model_metadata = yaml.safe_load(f)

            try:
                dr_client.delete_custom_model_by_git_model_id(
                    model_metadata[ModelSchema.MODEL_ID_KEY]
                )
            except (IllegalModelDeletion, DataRobotClientError):
                pass


def printout(msg):
    print(msg)


def unique_str():
    return f"{random.randint(1, 2 ** 32): 010}"


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
    def _setup_model(src_model_filepath, dst_models_root_dir, model_index):
        dst_model_path = dst_models_root_dir / f"model_{model_index}"
        shutil.copytree(src_model_filepath, dst_model_path)

        model_metadata_yaml_filepath = next(dst_model_path.rglob("**/model.yaml"))
        with open(model_metadata_yaml_filepath, "r") as reader:
            model_metadata = yaml.safe_load(reader)

        # Set model ID
        unique_string = unique_str()
        new_model_id = f"{model_metadata[ModelSchema.MODEL_ID_KEY]}-{unique_string}"
        ModelSchema.set_value(model_metadata, ModelSchema.MODEL_ID_KEY, value=new_model_id)

        # Set model name
        new_model_name = f"My Awsome GitHub Model {unique_string} [GitHub CI/CD, Functional Tests]"
        ModelSchema.set_value(
            model_metadata,
            ModelSchema.SETTINGS_SECTION_KEY,
            ModelSchema.NAME_KEY,
            value=new_model_name,
        )

        with open(model_metadata_yaml_filepath, "w") as writer:
            yaml.safe_dump(model_metadata, writer)

        return model_metadata

    # source model filepath
    src_models_root_dir = Path(__file__).parent / ".." / "models"
    src_model_path = next(
        p for p in src_models_root_dir.glob("**") if not p.samefile(src_models_root_dir)
    )
    dst_models_root_dir = repo_root_path / "models"

    first_model_metadata = _setup_model(src_model_path, dst_models_root_dir, 1)
    _setup_model(src_model_path, dst_models_root_dir, 2)

    # 7. Copy deployments
    deployments_src_root_dir = Path(__file__).parent / ".." / "deployments"
    dst_deployments_dir = repo_root_path / deployments_src_root_dir.name
    shutil.copytree(deployments_src_root_dir, dst_deployments_dir)

    deployment_yaml_filepath = next(dst_deployments_dir.glob("deployment.yaml"))
    with open(deployment_yaml_filepath, "r") as f:
        deployment_metadata = yaml.safe_load(f)
    DeploymentSchema.set_value(
        deployment_metadata,
        DeploymentSchema.MODEL_ID_KEY,
        value=first_model_metadata[ModelSchema.MODEL_ID_KEY],
    )
    with open(deployment_yaml_filepath, "w") as f:
        yaml.safe_dump(deployment_metadata, f)

    # 8. Add files to repo
    os.chdir(repo_root_path)
    git_repo.git.add("--all")
    git_repo.git.commit("-m", "Initial commit", "--no-verify")


@pytest.fixture
def set_model_dataset_for_testing(dr_client, model_metadata, model_metadata_yaml_file):
    if ModelSchema.TEST_KEY in model_metadata:
        test_dataset_filepath = (
            Path(__file__).parent
            / ".."
            / "datasets"
            / "juniors_3_year_stats_regression_pred_requests.csv"
        )
        dataset_id = None
        try:
            dataset_id = dr_client.upload_dataset(test_dataset_filepath)

            with temporarily_replace_schema_value(
                model_metadata_yaml_file,
                ModelSchema.TEST_KEY,
                ModelSchema.TEST_DATA_KEY,
                new_value=dataset_id,
            ):
                yield dataset_id
        finally:
            if dataset_id:
                dr_client.delete_dataset(dataset_id)
    else:
        yield


@contextlib.contextmanager
def _temporarily_replace_schema(yaml_filepath, *keys, metadata_or_value):
    try:
        origin_metadata = None
        with open(yaml_filepath) as f:
            origin_metadata = yaml.safe_load(f)

        if keys:  # Assuming a value replacement
            new_metadata = copy.deepcopy(origin_metadata)
            SharedSchema.set_value(new_metadata, *keys, value=metadata_or_value)
        else:  # Assuming a metadata replacement
            new_metadata = metadata_or_value

        with open(yaml_filepath, "w") as f:
            yaml.safe_dump(new_metadata, f)

        yield new_metadata

    finally:
        if origin_metadata:
            with open(yaml_filepath, "w") as f:
                yaml.safe_dump(origin_metadata, f)


@contextlib.contextmanager
def temporarily_replace_schema(yaml_filepath, new_metadata):
    with _temporarily_replace_schema(yaml_filepath, metadata_or_value=new_metadata) as new_metadata:
        yield new_metadata


@contextlib.contextmanager
def temporarily_replace_schema_value(yaml_filepath, key, *sub_keys, new_value):
    with _temporarily_replace_schema(
        yaml_filepath, key, *sub_keys, metadata_or_value=new_value
    ) as new_metadata:
        yield new_metadata


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
def feature_branch_name():
    return "feature"


@pytest.fixture
def merge_branch_name():
    return "merge-feature-branch"


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
    event_name,
    is_deploy,
    main_branch_head_sha=None,
    allow_deployment_deletion=True,
):
    main_branch_head_sha = main_branch_head_sha or git_repo.head.commit.hexsha
    ref_name = main_branch_name if event_name == "push" else "merge-branch"
    with github_env_set("GITHUB_EVENT_NAME", event_name), github_env_set(
        "GITHUB_SHA", git_repo.commit(main_branch_head_sha).hexsha
    ), github_env_set("GITHUB_BASE_REF", main_branch_name), github_env_set(
        "GITHUB_REF_NAME", ref_name
    ):
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


@contextlib.contextmanager
def upload_and_update_dataset(dr_client, dataset_filepath, metadata_yaml_filepath, *settings_keys):
    dataset_id = None
    try:
        dataset_id = dr_client.upload_dataset(dataset_filepath)
        with temporarily_replace_schema_value(
            metadata_yaml_filepath,
            SharedSchema.SETTINGS_SECTION_KEY,
            *settings_keys,
            new_value=dataset_id,
        ):
            yield dataset_id
    finally:
        if dataset_id:
            dr_client.delete_dataset(dataset_id)
