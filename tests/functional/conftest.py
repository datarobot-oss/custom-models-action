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
import re
import shutil
import socket
from collections import namedtuple
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml
from git import Repo

from common.convertors import MemoryConvertor
from common.exceptions import DataRobotClientError
from common.exceptions import IllegalModelDeletion
from common.github_env import GitHubEnv
from common.namepsace import Namespace
from dr_client import DrClient
from main import main
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from schema_validator import SharedSchema
from tests.conftest import unique_str

FUNCTIONAL_TESTS_NAMESPACE = "{}/datarobot/gh-action/functional-tests"
NUMBER_OF_MODELS_IN_TEST = 2


@pytest.fixture(name="github_repository_id", scope="session")
def github_repository_id_fixture():
    """
    A fixture to return the GitHub repository ID if exists. Otherwise, use the local machine
    name as the repository ID and return it.
    """

    unique_repository_id = GitHubEnv.repository_id()
    if not unique_repository_id:
        unique_repository_id = socket.gethostname()
        with github_env_set("GITHUB_REPOSITORY_ID", unique_repository_id):
            yield unique_repository_id
    else:
        yield unique_repository_id


# pylint: disable=unused-argument
@pytest.fixture(name="setup_functional_tests_namespace", scope="session")
def setup_functional_tests_namespace_fixture(github_repository_id):
    """
    A fixture to set up the GitHub action functional tests' namespace. Please note that this is
    required because there could be direct access to modules from the functional tests, such
    as the `dr_client`, `model_schema`, etc.
    """

    try:
        Namespace.init(FUNCTIONAL_TESTS_NAMESPACE.format(github_repository_id))
        yield
    finally:
        Namespace.uninit()


def create_dr_client():
    """Create a DataRobot client using parameter values from environment."""

    webserver = os.environ.get("DATAROBOT_WEBSERVER")
    api_token = os.environ.get("DATAROBOT_API_TOKEN")
    if webserver and api_token:
        return DrClient(webserver, api_token, verify_cert=False)
    return None


@pytest.fixture(name="dr_client", scope="session")
def fixture_dr_client(setup_functional_tests_namespace):
    """A fixture to create a DataRobot client."""

    return create_dr_client()


@lru_cache
def webserver_accessible():
    """Check if DataRobot web server is accessible."""

    dr_client = create_dr_client()
    return dr_client and dr_client.is_accessible()


# pylint: disable=unused-argument
@pytest.fixture(scope="session", autouse=True)
def setup_clean_datarobot_environment(setup_functional_tests_namespace, dr_client):
    """
    A fixture to delete deployments and custom models in DataRobot. There might be such remainders
    if a workflow is stopped in the middle.
    """

    if webserver_accessible():
        dr_client.delete_all_deployments(return_on_error=False)
        dr_client.delete_all_custom_models(return_on_error=False)


def cleanup_models(dr_client_tool, workspace_path):
    """Delete models in DataRobot, which are defined in the local repository source tree."""

    custom_models = dr_client_tool.fetch_custom_models()
    if custom_models:
        models_metadata = []
        for model_yaml_file in workspace_path.rglob("**/model.yaml"):
            with open(model_yaml_file, encoding="utf-8") as fd:
                raw_metadata = yaml.safe_load(fd)
                models_metadata.append(ModelSchema.validate_and_transform_single(raw_metadata))
        for model_yaml_file in workspace_path.rglob("**/models.yaml"):
            with open(model_yaml_file, encoding="utf-8") as fd:
                multi_models_metadata = ModelSchema.validate_and_transform_multi(yaml.safe_load(fd))
                for model_entry in multi_models_metadata[ModelSchema.MULTI_MODELS_KEY]:
                    models_metadata.append(model_entry[ModelSchema.MODEL_ENTRY_META_KEY])

        for model_metadata in models_metadata:
            try:
                dr_client_tool.delete_custom_model_by_user_provided_id(
                    model_metadata[ModelSchema.MODEL_ID_KEY]
                )
            except (IllegalModelDeletion, DataRobotClientError):
                pass


@pytest.fixture(name="sklearn_environment_drop_in_id", scope="session")
def fixture_sklearn_environment_drop_in_id(dr_client):
    """A fixture to fetch a sklearn environment drop-in to be used by the functional tests."""

    envs = dr_client.fetch_environment_drop_in()
    if len(envs) == 1:
        return envs[0]["id"]  # Assuming local dev env

    try:
        dr_sklearn_env_id = next(
            e["id"]
            for e in envs
            if re.search(r"[DataRobot] Python .* Scikit-Learn Drop-In", e["name"])
        )
        if dr_sklearn_env_id:
            return dr_sklearn_env_id
    except StopIteration:
        pass

    any_sklearn_env_id = next(
        e["id"] for e in envs if re.search(r"scikit-learn|sklearn|scikit", e["name"], re.I)
    )
    if any_sklearn_env_id:
        return any_sklearn_env_id

    assert False, "Scikit-Learn environment drop-in was not found in DR!"


def printout(msg):
    """A common print out method."""

    print(msg)


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


@pytest.fixture(name="workspace_path")
def fixture_workspace_path():
    """A fixture to create and return a temporary root dir to create a repository in it."""

    with TemporaryDirectory() as repo_tree:
        path = Path(repo_tree)
        yield path


@pytest.fixture(name="git_repo")
def fixture_git_repo(workspace_path):
    """
    A fixture to initialize a repository in a given root directory.

    Parameters
    ----------
    workspace_path : str
        The root folder for the source tree.
    """

    repo = Repo.init(workspace_path)
    repo.config_writer().set_value("user", "name", "functional-test-user").release()
    repo.config_writer().set_value("user", "email", "functional-test@company.com").release()
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    type(repo.git).GIT_PYTHON_TRACE = "full"
    return repo


# pylint: disable=too-many-statements
@pytest.fixture(name="build_repo_for_testing_factory")
def fixture_build_repo_for_testing_factory(
    workspace_path, git_repo, sklearn_environment_drop_in_id, github_repository_id
):
    """
    A fixture to build a complete source stree with model and deployment definitions in it. Then
    commit everything into the repository that was initialized in that root dir.
    """

    def _setup_model(
        src_model_filepath, dst_models_root_dir, model_index, dedicated_definition=True
    ):
        dst_model_dir_path = dst_models_root_dir / f"model_{model_index}"
        shutil.copytree(src_model_filepath, dst_model_dir_path)

        model_metadata_yaml_filepath = next(dst_model_dir_path.rglob("**/model.yaml"))
        with open(model_metadata_yaml_filepath, encoding="utf-8") as reader:
            model_metadata = yaml.safe_load(reader)

        # Set model ID
        namespace = FUNCTIONAL_TESTS_NAMESPACE.format(github_repository_id)
        origin_model_id = model_metadata[ModelSchema.MODEL_ID_KEY]
        unique_string = unique_str()
        new_model_id = f"{namespace}/{origin_model_id}-{model_index}-{unique_string}"
        ModelSchema.set_value(model_metadata, ModelSchema.MODEL_ID_KEY, value=new_model_id)

        # Set model name
        new_model_name = f"My Awsome GitHub Model {model_index} [GitHub CI/CD, Functional Tests]"
        ModelSchema.set_value(
            model_metadata,
            ModelSchema.SETTINGS_SECTION_KEY,
            ModelSchema.NAME_KEY,
            value=new_model_name,
        )

        # Set environment drop-in
        ModelSchema.set_value(
            model_metadata,
            ModelSchema.VERSION_KEY,
            ModelSchema.MODEL_ENV_ID_KEY,
            value=sklearn_environment_drop_in_id,
        )

        if dedicated_definition:
            save_new_metadata(model_metadata, model_metadata_yaml_filepath)
        else:
            # The definition will be written in a single multi-models definition.
            os.remove(model_metadata_yaml_filepath)

        return model_metadata, dst_model_dir_path

    def _setup_deployment(src_deployments_dir, dst_deployments_dir, models_meta):
        shutil.copytree(src_deployments_dir, dst_deployments_dir)

        deployment_yaml_filepath = next(dst_deployments_dir.glob("deployment.yaml"))
        with open(deployment_yaml_filepath, encoding="utf-8") as fd:
            deployment_metadata = yaml.safe_load(fd)

        # Set Deployment ID
        namespace = FUNCTIONAL_TESTS_NAMESPACE.format(github_repository_id)
        origin_deployment_id = deployment_metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]
        unique_string = unique_str()
        new_deployment_id = f"{namespace}/{origin_deployment_id}-{unique_string}"
        DeploymentSchema.set_value(
            deployment_metadata, DeploymentSchema.DEPLOYMENT_ID_KEY, value=new_deployment_id
        )

        # Set model ID
        DeploymentSchema.set_value(
            deployment_metadata,
            DeploymentSchema.MODEL_ID_KEY,
            value=models_meta[0].metadata[ModelSchema.MODEL_ID_KEY],
        )
        save_new_metadata(deployment_metadata, deployment_yaml_filepath)

    def _save_multi_models_metadata_yaml_file(
        dst_deployments_dir, models_meta, is_absolute_model_path, model_path_prefix
    ):
        multi_models_definition = {ModelSchema.MULTI_MODELS_KEY: []}
        for model_meta in models_meta:
            relative_model_dir_path = os.path.relpath(
                model_meta.dst_model_dir_path, dst_deployments_dir
            )
            if is_absolute_model_path:
                model_path = relative_model_dir_path.replace("../", model_path_prefix)
            else:
                model_path = relative_model_dir_path
            multi_models_definition[ModelSchema.MULTI_MODELS_KEY].append(
                {
                    ModelSchema.MODEL_ENTRY_META_KEY: model_meta.metadata,
                    ModelSchema.MODEL_ENTRY_PATH_KEY: model_path,
                }
            )
        save_new_metadata(multi_models_definition, dst_deployments_dir / "models.yaml")

    def _inner(dedicated_model_definition, is_absolute_model_path=False, model_path_prefix=None):
        src_models_root_dir = Path(__file__).parent / ".." / "models"
        src_model_path = next(
            p for p in src_models_root_dir.glob("**") if not p.samefile(src_models_root_dir)
        )
        dst_models_root_dir = workspace_path / "models"

        ModelMeta = namedtuple("ModelMeta", ["metadata", "dst_model_dir_path"])
        models_meta = []
        for counter in range(1, 1 + NUMBER_OF_MODELS_IN_TEST):
            metadata, dst_model_dir_path = _setup_model(
                src_model_path, dst_models_root_dir, counter, dedicated_model_definition
            )
            models_meta.append(ModelMeta(metadata, dst_model_dir_path))

        src_deployments_dir = Path(__file__).parent / ".." / "deployments"
        dst_deployments_dir = workspace_path / src_deployments_dir.name
        _setup_deployment(src_deployments_dir, dst_deployments_dir, models_meta)

        if not dedicated_model_definition:
            _save_multi_models_metadata_yaml_file(
                dst_deployments_dir, models_meta, is_absolute_model_path, model_path_prefix
            )

        # 8. Add files to repo in multiple commits in order to avoid a use case of too few commits
        #    that can be regarded as not a merge branch
        os.chdir(workspace_path)
        git_repo.git.add("--all")
        git_repo.git.commit("-m", "Initial commit", "--no-verify")

    return _inner


@pytest.fixture(name="build_repo_for_testing")
def fixture_build_repo_for_testing(build_repo_for_testing_factory):
    """
    A fixture to build a complete source stree with model and deployment definitions in it. Each
    model has its own metadata definition in a dedicated YAML file. Then commit everything into
    the repository that was initialized in that root dir.
    """

    build_repo_for_testing_factory(dedicated_model_definition=True)


@pytest.fixture
def set_model_dataset_for_testing(dr_client, git_repo, model_metadata, model_metadata_yaml_file):
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
            ), temporarily_replace_metadata_value(
                ModelSchema,
                model_metadata,
                ModelSchema.TEST_KEY,
                ModelSchema.TEST_DATA_ID_KEY,
                new_value=dataset_id,
            ):
                git_commit_metadata_yaml_file(
                    git_repo, model_metadata_yaml_file, "Update model's test dataset"
                )
                yield dataset_id
        finally:
            if dataset_id:
                git_commit_metadata_yaml_file(
                    git_repo, model_metadata_yaml_file, "Update model's test dataset"
                )
                dr_client.delete_dataset(dataset_id)
    else:
        yield


def git_commit_metadata_yaml_file(git_repo, yaml_file_path, message):
    """Commit changes to a metadata YAML file."""

    git_repo.git.add(yaml_file_path)
    git_repo.git.commit("-m", message)


def save_new_metadata(metadata, yaml_file_path):
    """Save a new model/deployment metadata."""

    with open(yaml_file_path, "w", encoding="utf-8") as fd:
        yaml.safe_dump(metadata, fd)


def save_new_metadata_and_commit(metadata, yaml_file_path, git_repo, message):
    """Save a new model/deployment metadata and commit with a given message."""

    save_new_metadata(metadata, yaml_file_path)
    git_commit_metadata_yaml_file(git_repo, yaml_file_path, message)


@contextlib.contextmanager
def temporarily_replace_metadata_value(schema_cls, metadata, key, *sub_keys, new_value):
    """A context manager to temporarily replace a value in a given metadata of a given schema."""

    origin_value = None
    try:
        origin_value = schema_cls.get_value(metadata, key, *sub_keys)
        schema_cls.set_value(metadata, key, *sub_keys, value=new_value)
        yield
    finally:
        schema_cls.set_value(metadata, key, *sub_keys, value=origin_value)


@pytest.fixture
def set_deployment_actuals_dataset(
    dr_client, git_repo, deployment_metadata, deployment_metadata_yaml_file
):
    """
    A fixture to temporarily upload and set a deployment's dataset in a deployment definition and
    DataRobot.
    """

    actuals_dataset_id = DeploymentSchema.get_value(
        deployment_metadata,
        DeploymentSchema.SETTINGS_SECTION_KEY,
        DeploymentSchema.ASSOCIATION_KEY,
        DeploymentSchema.ASSOCIATION_ACTUALS_DATASET_ID_KEY,
    )
    if actuals_dataset_id:
        actuals_dataset_filepath = (
            Path(__file__).parent
            / ".."
            / "datasets"
            / "juniors_3_year_stats_regression_actuals.csv"
        )
        dataset_id = None
        try:
            dataset_id = dr_client.upload_dataset(actuals_dataset_filepath)

            with temporarily_replace_schema_value(
                deployment_metadata_yaml_file,
                DeploymentSchema.SETTINGS_SECTION_KEY,
                DeploymentSchema.ASSOCIATION_KEY,
                DeploymentSchema.ASSOCIATION_ACTUALS_DATASET_ID_KEY,
                new_value=dataset_id,
            ), temporarily_replace_metadata_value(
                DeploymentSchema,
                deployment_metadata,
                DeploymentSchema.SETTINGS_SECTION_KEY,
                DeploymentSchema.ASSOCIATION_KEY,
                DeploymentSchema.ASSOCIATION_ACTUALS_DATASET_ID_KEY,
                new_value=dataset_id,
            ):
                git_commit_metadata_yaml_file(
                    git_repo, deployment_metadata_yaml_file, "Update deployment's actuals dataset"
                )
                yield dataset_id
        finally:
            if dataset_id:
                git_commit_metadata_yaml_file(
                    git_repo, deployment_metadata_yaml_file, "Update deployment's actuals dataset"
                )
                dr_client.delete_dataset(dataset_id)
    else:
        yield


def replace_and_store_schema(yaml_filepath, *keys, metadata_or_value):
    """
    A helper method to replace and save either a provided complete metadata or a single value
    of a metadata.
    """

    with open(yaml_filepath, encoding="utf-8") as fd:
        origin_metadata = yaml.safe_load(fd)

    if keys:  # Assuming a value replacement
        new_metadata = copy.deepcopy(origin_metadata)
        SharedSchema.set_value(new_metadata, *keys, value=metadata_or_value)
    else:  # Assuming a metadata replacement
        new_metadata = metadata_or_value

    save_new_metadata(new_metadata, yaml_filepath)
    return origin_metadata, new_metadata


@contextlib.contextmanager
def _temporarily_replace_schema(yaml_filepath, *keys, metadata_or_value):
    origin_metadata = None
    try:
        origin_metadata, new_metadata = replace_and_store_schema(
            yaml_filepath, *keys, metadata_or_value=metadata_or_value
        )
        yield new_metadata
    finally:
        if origin_metadata:
            save_new_metadata(origin_metadata, yaml_filepath)


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
def skip_model_testing(git_repo, numbered_model_metadata, numbered_model_metadata_yaml_file):
    """A fixture to skip model testing in DataRobot."""

    origin_test_section = []
    for model_number in range(1, 1 + NUMBER_OF_MODELS_IN_TEST):
        model_metadata = numbered_model_metadata(model_number)
        origin_test_section.append(model_metadata.get(ModelSchema.TEST_KEY))
        model_metadata.pop(ModelSchema.TEST_KEY, None)
        model_metadata_yaml_file = numbered_model_metadata_yaml_file(model_number)
        save_new_metadata(model_metadata, model_metadata_yaml_file)
        git_repo.git.add(model_metadata_yaml_file)

    git_repo.git.commit("-m", "Skip model(s) testing")

    yield

    try:
        for model_number in range(1, 1 + NUMBER_OF_MODELS_IN_TEST):
            model_metadata = numbered_model_metadata(model_number)
            model_metadata[ModelSchema.TEST_KEY] = origin_test_section.pop(0)
            model_metadata_yaml_file = numbered_model_metadata_yaml_file(model_number)
            save_new_metadata(model_metadata, model_metadata_yaml_file)
            git_repo.git.add(model_metadata_yaml_file)

        git_repo.git.commit("-m", "Restore model(s) testing section")
    except StopIteration:
        # It's a kind of best-effort operation. The test may change the models' hierarchy.
        pass


# NOTE: it was rather better to use the pytest.mark.usefixture for 'build_repo_for_testing'
# but, it turns out that it doesn't work with fixtures.
@pytest.fixture(name="numbered_model_metadata_yaml_file")
def fixture_numbered_model_metadata_yaml_file(build_repo_for_testing, workspace_path, git_repo):
    """A fixture to load and return a given numbered model in the local source tree."""
    # pylint: disable=unused-argument

    def _inner(model_number):
        return next(workspace_path.rglob(f"*_{model_number}/model.yaml"))

    return _inner


@pytest.fixture(name="model_metadata_yaml_file")
def fixture_model_metadata_yaml_file(numbered_model_metadata_yaml_file):
    """A fixture to load and return the first defined model in the local source tree."""

    return numbered_model_metadata_yaml_file(model_number=1)


@pytest.fixture(name="numbered_model_metadata")
def fixture_numbered_model_metadata(numbered_model_metadata_yaml_file):
    """A fixture to load and return given numbered model metadata from a given yaml definition."""

    def _inner(model_number):
        with open(numbered_model_metadata_yaml_file(model_number), encoding="utf-8") as fd:
            raw_metadata = yaml.safe_load(fd)
            return ModelSchema.validate_and_transform_single(raw_metadata)

    return _inner


@pytest.fixture(name="model_metadata")
def fixture_model_metadata(numbered_model_metadata):
    """A fixture to load and return model metadata from a given yaml definition."""

    return numbered_model_metadata(model_number=1)


@pytest.fixture(name="deployment_metadata_yaml_file")
def fixture_deployment_metadata_yaml_file(workspace_path, build_repo_for_testing):
    """A fixture to return a unique deployment from the temporary created local source tree."""

    return next(workspace_path.rglob("**/deployment.yaml"))


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


def increase_model_memory_by_1mb(git_repo, model_yaml_file, do_commit=True):
    """A method to increase a model's memory in a model definition and save it locally."""

    with open(model_yaml_file, encoding="utf-8") as fd:
        yaml_content = yaml.safe_load(fd)
        memory = ModelSchema.get_value(
            yaml_content, ModelSchema.VERSION_KEY, ModelSchema.MEMORY_KEY
        )
        memory = memory or "2048Mi"
        num_part, unit = MemoryConvertor._extract_unit_fields(memory)
        new_memory = f"{num_part+1}{unit}"
        yaml_content[ModelSchema.VERSION_KEY][ModelSchema.MEMORY_KEY] = new_memory

    if do_commit:
        save_new_metadata_and_commit(
            yaml_content, model_yaml_file, git_repo, f"Increase model memory by {memory}"
        )
    else:
        save_new_metadata(yaml_content, model_yaml_file)

    return new_memory


def run_github_action(
    workspace_path,
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
    workspace_path : str or pathlib.Path
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
    with github_env_set("GITHUB_WORKSPACE", str(workspace_path)), github_env_set(
        "GITHUB_SHA", git_repo.commit(main_branch_head_sha).hexsha
    ), github_env_set("GITHUB_EVENT_NAME", event_name), github_env_set(
        "GITHUB_SHA", git_repo.commit(main_branch_head_sha).hexsha
    ), github_env_set(
        "GITHUB_BASE_REF", main_branch_name
    ), github_env_set(
        "GITHUB_REF_NAME", ref_name
    ):
        datarobot_webserver = os.environ.get("DATAROBOT_WEBSERVER")
        namespace = FUNCTIONAL_TESTS_NAMESPACE.format(GitHubEnv.repository_id())
        args = [
            "--webserver",
            datarobot_webserver,
            "--api-token",
            os.environ.get("DATAROBOT_API_TOKEN"),
            "--branch",
            main_branch_name,
            "--namespace",
            namespace,
            "--allow-model-deletion",
        ]

        if not is_deploy:
            args.append("--models-only")

        if allow_deployment_deletion:
            args.append("--allow-deployment-deletion")

        if not any(
            webserver_with_cert in datarobot_webserver
            for webserver_with_cert in ["https://app.datarobot.com", "https://app.eu.datarobot.com"]
        ):
            args.append("--skip-cert-verification")

        main(args)


@contextlib.contextmanager
def upload_and_update_dataset(
    dr_client, dataset_filepath, metadata_yaml_filepath, section_key, *sub_keys
):
    """
    Upload and update a dataset in a settings section, then yield. Upon return, it deletes the
    dataset from DataRobot.
    """

    dataset_id = None
    try:
        dataset_id = dr_client.upload_dataset(dataset_filepath)
        with temporarily_replace_schema_value(
            metadata_yaml_filepath,
            section_key,
            *sub_keys,
            new_value=dataset_id,
        ):
            yield dataset_id
    finally:
        if dataset_id:
            dr_client.delete_dataset(dataset_id)


@contextlib.contextmanager
def temporarily_upload_training_dataset_for_structured_model(
    dr_client, model_metadata_yaml_file, is_model_level, event_name
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
        section_key = (
            ModelSchema.SETTINGS_SECTION_KEY if is_model_level else ModelSchema.VERSION_KEY
        )
        with upload_and_update_dataset(
            dr_client,
            training_and_holdout_dataset_filepath,
            model_metadata_yaml_file,
            section_key,
            ModelSchema.TRAINING_DATASET_ID_KEY,
        ) as training_dataset_id:
            partition_column = "partitioning"
            with temporarily_replace_schema_value(
                model_metadata_yaml_file,
                section_key,
                ModelSchema.PARTITIONING_COLUMN_KEY,
                new_value=partition_column,
            ):
                yield training_dataset_id, partition_column


@pytest.fixture
def cleanup(dr_client, workspace_path):
    """A fixture to delete models in DataRobot that were created from the local source tree."""

    yield

    cleanup_models(dr_client, workspace_path)
