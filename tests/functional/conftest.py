#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=protected-access
# pylint: disable=too-many-arguments

"""A functional test configuration module."""

import contextlib
import copy
import logging
import os
import random
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
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from schema_validator import SharedSchema


def webserver_accessible():
    """Check if DataRobot web server is accessible."""

    webserver = os.environ.get("DATAROBOT_WEBSERVER")
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    if webserver and api_token:
        return DrClient(webserver, api_token, verify_cert=False).is_accessible()
    return False


def cleanup_models(dr_client_tool, repo_root_path):
    """Delete models in DataRobot, which are defined in the local repository source tree."""

    custom_models = dr_client_tool.fetch_custom_models()
    if custom_models:
        for model_yaml_file in repo_root_path.rglob("**/model.yaml"):
            with open(model_yaml_file, encoding="utf-8") as fd:
                model_metadata = yaml.safe_load(fd)

            try:
                dr_client_tool.delete_custom_model_by_user_provided_id(
                    model_metadata[ModelSchema.MODEL_ID_KEY]
                )
            except (IllegalModelDeletion, DataRobotClientError):
                pass


def printout(msg):
    """A common print out method."""

    print(msg)


def unique_str():
    """Generate a unique 10-chars long string."""

    return f"{random.randint(1, 2 ** 32): 010}"


@contextlib.contextmanager
def github_env_set(env_key, env_value):
    """
    Set environment variable to simulate GitHub actions environment.

    Parameters
    ----------
    env_key : str
        Environment variable key.
    env_value : str
        Environment variable value.
    """

    does_exist = env_key in os.environ
    if does_exist:
        old_value = os.environ[env_key]
    os.environ[env_key] = env_value
    yield
    if does_exist:
        os.environ[env_key] = old_value


@pytest.fixture(name="repo_root_path")
def fixture_repo_root_path():
    """A fixture to create and return a temporary root dir to create a repository in it."""

    with TemporaryDirectory() as repo_tree:
        path = Path(repo_tree)
        yield path


@pytest.fixture(name="git_repo")
def fixture_git_repo(repo_root_path):
    """
    A fixture to initialize a repository in a given root directory.

    Parameters
    ----------
    repo_root_path : str
        The root folder for the source tree.
    """

    repo = Repo.init(repo_root_path)
    repo.config_writer().set_value("user", "name", "functional-test-user").release()
    repo.config_writer().set_value("user", "email", "functional-test@company.com").release()
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    type(repo.git).GIT_PYTHON_TRACE = "full"
    return repo


@pytest.fixture(name="build_repo_for_testing")
def fixture_build_repo_for_testing(repo_root_path, git_repo):
    """
    A fixture to build a complete source stree with model and deployment definitions in it. Then
    commit everything into the repository that was initialized in that root dir.
    """

    def _setup_model(src_model_filepath, dst_models_root_dir, model_index):
        dst_model_path = dst_models_root_dir / f"model_{model_index}"
        shutil.copytree(src_model_filepath, dst_model_path)

        model_metadata_yaml_filepath = next(dst_model_path.rglob("**/model.yaml"))
        with open(model_metadata_yaml_filepath, encoding="utf-8") as reader:
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

        with open(model_metadata_yaml_filepath, "w", encoding="utf-8") as writer:
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
    with open(deployment_yaml_filepath, encoding="utf-8") as fd:
        deployment_metadata = yaml.safe_load(fd)
    DeploymentSchema.set_value(
        deployment_metadata,
        DeploymentSchema.MODEL_ID_KEY,
        value=first_model_metadata[ModelSchema.MODEL_ID_KEY],
    )
    with open(deployment_yaml_filepath, "w", encoding="utf-8") as fd:
        yaml.safe_dump(deployment_metadata, fd)

    # 8. Add files to repo
    os.chdir(repo_root_path)
    git_repo.git.add("--all")
    git_repo.git.commit("-m", "Initial commit", "--no-verify")


@pytest.fixture
def set_model_dataset_for_testing(dr_client, model_metadata, model_metadata_yaml_file):
    """
    A fixture to temporarily upload and set a model's dataset in a model definition and DataRobot.
    """

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
                ModelSchema.TEST_DATA_ID_KEY,
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
        with open(yaml_filepath, encoding="utf-8") as fd:
            origin_metadata = yaml.safe_load(fd)

        if keys:  # Assuming a value replacement
            new_metadata = copy.deepcopy(origin_metadata)
            SharedSchema.set_value(new_metadata, *keys, value=metadata_or_value)
        else:  # Assuming a metadata replacement
            new_metadata = metadata_or_value

        with open(yaml_filepath, "w", encoding="utf-8") as fd:
            yaml.safe_dump(new_metadata, fd)

        yield new_metadata

    finally:
        if origin_metadata:
            with open(yaml_filepath, "w", encoding="utf-8") as fd:
                yaml.safe_dump(origin_metadata, fd)


@contextlib.contextmanager
def temporarily_replace_schema(yaml_filepath, new_metadata):
    """Temporarily replace a metadata if a given yaml definition."""

    with _temporarily_replace_schema(yaml_filepath, metadata_or_value=new_metadata) as metadata:
        yield metadata


@contextlib.contextmanager
def temporarily_replace_schema_value(yaml_filepath, key, *sub_keys, new_value):
    """Temporarily replace a value in a given metadata and yaml definition."""

    with _temporarily_replace_schema(
        yaml_filepath, key, *sub_keys, metadata_or_value=new_value
    ) as metadata:
        yield metadata


@pytest.fixture
def skip_model_testing(model_metadata, model_metadata_yaml_file):
    """A fixture to skip model testing in DataRobot."""

    origin_test_section = model_metadata.get(ModelSchema.TEST_KEY)
    model_metadata.pop(ModelSchema.TEST_KEY, None)
    with open(model_metadata_yaml_file, "w", encoding="utf-8") as fd:
        yaml.safe_dump(model_metadata, fd)

    yield

    model_metadata[ModelSchema.TEST_KEY] = origin_test_section
    with open(model_metadata_yaml_file, "w", encoding="utf-8") as fd:
        yaml.safe_dump(model_metadata, fd)


# NOTE: it was rather better to use the pytest.mark.usefixture for 'build_repo_for_testing'
# but, apparently it cannot be used with fixtures.
@pytest.fixture(name="model_metadata_yaml_file")
def fixture_model_metadata_yaml_file(build_repo_for_testing, repo_root_path, git_repo):
    """A fixture to load and return the first defined model in the local source tree."""

    # pylint: disable=unused-argument

    model_yaml_file = next(repo_root_path.rglob("*_1/model.yaml"))
    with open(model_yaml_file, encoding="utf-8") as fd:
        yaml_content = yaml.safe_load(fd)
        yaml_content[ModelSchema.MODEL_ID_KEY] = f"my-awesome-model-{str(ObjectId())}"

    with open(model_yaml_file, "w", encoding="utf-8") as fd:
        yaml.safe_dump(yaml_content, fd)

    git_repo.git.add(model_yaml_file)
    git_repo.git.commit("--amend", "--no-edit")

    return model_yaml_file


@pytest.fixture(name="model_metadata")
def fixture_model_metadata(model_metadata_yaml_file):
    """A fixture to load and return model metadata from a given yaml definition."""

    with open(model_metadata_yaml_file, encoding="utf-8") as fd:
        return yaml.safe_load(fd)


@pytest.fixture(name="main_branch_name")
def fixture_main_branch_name():
    """A fixture to return the main branch name."""

    return "master"


@pytest.fixture
def feature_branch_name():
    """A fixture to return the feature branch name."""

    return "feature"


@pytest.fixture
def merge_branch_name():
    """A fixture to return the merge branch name."""

    return "merge-feature-branch"


@pytest.fixture(name="dr_client")
def fixture_dr_client():
    """A fixture to create a DataRobot client."""

    webserver = os.environ.get("DATAROBOT_WEBSERVER")
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    return DrClient(webserver, api_token, verify_cert=False)


def increase_model_memory_by_1mb(model_yaml_file):
    """A method to increase a model's memory in a model definition and save it locally."""

    with open(model_yaml_file, encoding="utf-8") as fd:
        yaml_content = yaml.safe_load(fd)
        memory = ModelSchema.get_value(
            yaml_content, ModelSchema.VERSION_KEY, ModelSchema.MEMORY_KEY
        )
        memory = memory or "256Mi"
        num_part, unit = MemoryConvertor._extract_unit_fields(memory)
        new_memory = f"{num_part+1}{unit}"
        yaml_content[ModelSchema.VERSION_KEY][ModelSchema.MEMORY_KEY] = new_memory

    with open(model_yaml_file, "w", encoding="utf-8") as fd:
        yaml.safe_dump(yaml_content, fd)

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
    """
    Execute a GitHub action. The `is_deploy` attribute determines whether it is a custom
    inference model action or a deployment action.

    Parameters
    ----------
    repo_root_path : str
        The repository root directory.
    git_repo : git.Repo
        A tool to interact with the local Git repository.
    main_branch_name : str
        The main branch name.
    event_name : str
        The GitHub event name that triggers the workflow.
    is_deploy : bool
        Whether it's a deployment or a custom inference model actions.
    main_branch_head_sha : str, optional
        The main branch HEAD SHA.
    allow_deployment_deletion : bool, optional
        Whether to allow a deployment deletion in DataRobot by the executed action.
    """

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
    """
    Upload and update a dataset in a settings section, then yield. Upon return, it deletes the
    dataset from DataRobot.
    """

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


@contextlib.contextmanager
def temporarily_upload_training_dataset_for_structured_model(
    dr_client, model_metadata_yaml_file, event_name="pull_request"
):
    """A method to temporarily upload a training dataset for a structured model."""

    if event_name == "pull_request":
        yield None, None
    else:
        printout("Upload training with holdout dataset for structured model.")
        datasets_root = Path(__file__).parent / ".." / "datasets"
        training_and_holdout_dataset_filepath = (
            datasets_root / "juniors_3_year_stats_regression_structured_training_with_holdout.csv"
        )
        with upload_and_update_dataset(
            dr_client,
            training_and_holdout_dataset_filepath,
            model_metadata_yaml_file,
            ModelSchema.TRAINING_DATASET_ID_KEY,
        ) as training_dataset_id:
            partition_column = "partitioning"
            with temporarily_replace_schema_value(
                model_metadata_yaml_file,
                ModelSchema.SETTINGS_SECTION_KEY,
                ModelSchema.PARTITIONING_COLUMN_KEY,
                new_value=partition_column,
            ):
                yield training_dataset_id, partition_column
