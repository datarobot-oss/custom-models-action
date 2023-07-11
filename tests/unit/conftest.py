#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=too-many-arguments

"""A configuration test module for unit-tests."""
import argparse
import contextlib
import logging
import os
import re
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import yaml
from bson import ObjectId
from git import Repo

from common import constants
from common.namepsace import Namespace
from custom_models_action import CustomModelsAction
from metrics import Metric
from metrics import Metrics
from model_controller import ModelController
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema


@pytest.fixture(name="setup_unit_tests_namespace", scope="session", autouse=True)
def fixture_setup_unit_tests_namespace():
    """A fixture to setup default unit-tests namespace"""

    with patch.dict(os.environ, {"GITHUB_REPOSITORY_ID": "1234567"}):
        Namespace.init("datarobot/gh-unit-tests")
        yield


@pytest.fixture(name="workspace_path")
def fixture_workspace_path():
    """
    A fixture to temporarily create and return a workspace folder in which a repository will be
    initialized.
    """

    with TemporaryDirectory() as repo_tree:
        yield Path(repo_tree)


def write_to_file(file_path, content):
    """A method to write into a file."""

    with open(file_path, "w", encoding="utf-8") as fd:
        fd.write(content)


def make_a_change_and_commit(git_repo, file_paths, index):
    """Makes a single change in a provided Python files and commit the changes."""

    for file_path in file_paths:
        with open(file_path, "a", encoding="utf-8") as fd:
            fd.write(f"\n# Automatic change ({index})")
    git_repo.index.add([str(f) for f in file_paths])
    git_repo.index.commit(f"Change number {index}")


def create_partial_model_schema(is_single=True, num_models=1, with_target_type=False):
    """Creates a partial model schema in a single/multi model form definitions."""

    def _partial_model_schema(name):
        partial_schema = {
            ModelSchema.MODEL_ID_KEY: str(uuid.uuid4()),
            ModelSchema.SETTINGS_SECTION_KEY: {
                ModelSchema.NAME_KEY: name,
                ModelSchema.TARGET_NAME_KEY: "target_feature_col",
            },
            ModelSchema.VERSION_KEY: {ModelSchema.MODEL_ENV_ID_KEY: str(ObjectId())},
        }
        if with_target_type:
            partial_schema[ModelSchema.TARGET_TYPE_KEY] = ModelSchema.TARGET_TYPE_REGRESSION
        return partial_schema

    if is_single:
        model_schema = _partial_model_schema("single-model")
    else:
        model_schema = {ModelSchema.MULTI_MODELS_KEY: []}
        for counter in range(1, num_models + 1):
            model_name = f"model-{counter}"
            model_schema[ModelSchema.MULTI_MODELS_KEY].append(
                {
                    ModelSchema.MODEL_ENTRY_PATH_KEY: f"./path/to/{model_name}",
                    ModelSchema.MODEL_ENTRY_META_KEY: _partial_model_schema(model_name),
                }
            )
    return model_schema


def create_partial_deployment_schema(is_single=True, num_deployments=1):
    """Creates a partial deployment schema in a single/multi form deployment definition."""

    def _partial_deployment_schema(name):
        return {
            DeploymentSchema.DEPLOYMENT_ID_KEY: str(uuid.uuid4()),
            DeploymentSchema.MODEL_ID_KEY: str(uuid.uuid4()),
            DeploymentSchema.SETTINGS_SECTION_KEY: {
                DeploymentSchema.LABEL_KEY: name,
            },
        }

    if is_single:
        return _partial_deployment_schema("single-deployment")

    deployments_schema = []
    for counter in range(1, num_deployments + 1):
        single_deployment_schema = _partial_deployment_schema(f"deployment-{counter}")
        deployments_schema.append(single_deployment_schema)
    return deployments_schema


@pytest.fixture(name="common_path")
def fixture_common_path(workspace_path):
    """A fixture that returns the common directory path from the repository root."""

    return workspace_path / "common"


@pytest.fixture(name="common_filepath")
def fixture_common_filepath(common_path):
    """A fixture that returns the common.py file path under the common path."""

    return common_path / "common.py"


# pylint: disable=unused-argument
@pytest.fixture(name="common_path_with_code")
def fixture_common_path_with_code(workspace_path, common_path, common_filepath):
    """
    A fixture to create a common path under the repository root dire and occupies it with
    source code.
    """

    os.makedirs(common_path)
    write_to_file(common_filepath, "# common.py")
    write_to_file(common_path / "util.py", "# Util.py")
    os.makedirs(common_path / "string")
    write_to_file(common_path / "string" / "conv.py", "# conv.py")
    return common_path


@pytest.fixture(name="excluded_src_path")
def fixture_excluded_src_path(workspace_path):
    """
    A fixture to create directory and file under the root repository path that will not be
    part of any model definition.
    """

    excluded_path = workspace_path / "excluded_path"
    os.makedirs(excluded_path)
    write_to_file(excluded_path / "some_file.py", "# some_file.py")
    return excluded_path


@contextlib.contextmanager
def _un_namespaced_model_user_provided_id(single_or_multi_model_metadata):
    if ModelSchema.is_multi_models_schema(single_or_multi_model_metadata):
        if not single_or_multi_model_metadata[ModelSchema.MULTI_MODELS_KEY]:
            yield
        else:
            for model_entry in single_or_multi_model_metadata[ModelSchema.MULTI_MODELS_KEY]:
                model_metadata = model_entry[ModelSchema.MODEL_ENTRY_META_KEY]
                model_metadata[ModelSchema.MODEL_ID_KEY] = Namespace.un_namespaced(
                    model_metadata[ModelSchema.MODEL_ID_KEY]
                )
            try:
                yield
            finally:
                for model_entry in single_or_multi_model_metadata[ModelSchema.MULTI_MODELS_KEY]:
                    model_metadata = model_entry[ModelSchema.MODEL_ENTRY_META_KEY]
                    model_metadata[ModelSchema.MODEL_ID_KEY] = Namespace.namespaced(
                        model_metadata[ModelSchema.MODEL_ID_KEY]
                    )
    else:
        single_or_multi_model_metadata[ModelSchema.MODEL_ID_KEY] = Namespace.un_namespaced(
            single_or_multi_model_metadata[ModelSchema.MODEL_ID_KEY]
        )
        try:
            yield
        finally:
            single_or_multi_model_metadata[ModelSchema.MODEL_ID_KEY] = Namespace.namespaced(
                single_or_multi_model_metadata[ModelSchema.MODEL_ID_KEY]
            )


@pytest.fixture(name="single_model_factory")
def fixture_single_model_factory(workspace_path, common_path_with_code):
    """A factory fixture to create a single model definition."""

    def _inner(
        name,
        write_metadata=True,
        with_include_glob=True,
        with_exclude_glob=True,
        include_main_prog=True,
        user_provided_id=None,
    ):
        model_path = workspace_path / name
        os.makedirs(model_path)
        if include_main_prog:
            write_to_file(model_path / "custom.py", "# custom.py")
        write_to_file(model_path / "README.md", "# README")
        write_to_file(model_path / "non-datarobot-yaml.yaml", '{"models": []}')
        os.makedirs(model_path / "score")
        write_to_file(model_path / "score" / "score.py", "# score.py")

        single_model_metadata = {
            ModelSchema.MODEL_ID_KEY: Namespace.namespaced(user_provided_id or str(uuid.uuid4())),
            ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION,
            ModelSchema.SETTINGS_SECTION_KEY: {
                ModelSchema.NAME_KEY: name,
                ModelSchema.TARGET_NAME_KEY: "Grade 2014",
            },
            ModelSchema.VERSION_KEY: {ModelSchema.MODEL_ENV_ID_KEY: str(ObjectId())},
        }
        if with_include_glob:
            # noinspection PyTypeChecker
            single_model_metadata[ModelSchema.VERSION_KEY][ModelSchema.INCLUDE_GLOB_KEY] = [
                "./",
                f"/{common_path_with_code.relative_to(workspace_path)}/",
            ]
        if with_exclude_glob:
            # noinspection PyTypeChecker
            single_model_metadata[ModelSchema.VERSION_KEY][ModelSchema.EXCLUDE_GLOB_KEY] = [
                "./README.md"
            ]

        metadata_yaml_filepath = None
        if write_metadata:
            metadata_yaml_filepath = model_path / "model.yaml"
            with _un_namespaced_model_user_provided_id(single_model_metadata):
                write_to_file(metadata_yaml_filepath, yaml.dump(single_model_metadata))

        return single_model_metadata, metadata_yaml_filepath

    yield _inner


# pylint: disable=unused-argument
@pytest.fixture(name="models_factory")
def fixture_models_factory(common_path_with_code, workspace_path, single_model_factory):
    """A fixture to create multiple model definitions."""

    def _inner(
        num_models=2,
        is_multi=False,
        with_include_glob=True,
        with_exclude_glob=True,
        include_main_prog=True,
        is_absolute_path=False,
        root_prefix=None,
    ):
        models_metadata = []
        for counter in range(num_models):
            model_metadata, _ = single_model_factory(
                name=f"model_multi_{counter}" if is_multi else f"model_{counter}",
                write_metadata=not is_multi,
                with_include_glob=with_include_glob,
                with_exclude_glob=with_exclude_glob,
                include_main_prog=include_main_prog,
            )
            models_metadata.append(model_metadata)

        if is_multi:
            multi_models_yaml_content = {ModelSchema.MULTI_MODELS_KEY: []}
            for model_metadata in models_metadata:
                name = ModelSchema.get_value(
                    model_metadata, ModelSchema.SETTINGS_SECTION_KEY, ModelSchema.NAME_KEY
                )
                if is_absolute_path:
                    model_path = f"{root_prefix}{name}"
                else:
                    model_path = f"../{name}"
                multi_models_yaml_content[ModelSchema.MULTI_MODELS_KEY].append(
                    {
                        ModelSchema.MODEL_ENTRY_PATH_KEY: model_path,
                        ModelSchema.MODEL_ENTRY_META_KEY: model_metadata,
                    }
                )

            with _un_namespaced_model_user_provided_id(multi_models_yaml_content):
                multi_models_content = yaml.dump(multi_models_yaml_content)
            models_metadata_dir_path = workspace_path / "metadata"
            if not os.path.exists(models_metadata_dir_path):
                os.makedirs(models_metadata_dir_path)
            write_to_file(models_metadata_dir_path / "models.yaml", multi_models_content)

        return models_metadata

    return _inner


@pytest.fixture(name="single_model_root_path")
def fixture_single_model_root_path(workspace_path):
    """A fixture to return the first model root path."""

    return workspace_path / "model_0"


# pylint: disable=unused-argument
@pytest.fixture(name="single_model_file_paths")
def fixture_single_model_file_paths(workspace_path, models_factory, single_model_root_path):
    """A fixture to return all the file paths below to a just created model."""

    models_factory(1)
    return list(single_model_root_path.rglob("*.*"))


@pytest.fixture
def options(workspace_path):
    """A fixture to mock a parse args namespace options."""

    with patch.dict(os.environ, {"GITHUB_WORKSPACE": str(workspace_path)}):
        yield argparse.Namespace(
            webserver="http://www.dummy.com",
            api_token="abc123",
            branch="master",
            allow_model_deletion=True,
            allow_deployment_deletion=True,
            skip_cert_verification=True,
            models_only=False,
        )


@pytest.fixture
def mock_prerequisites():
    """A fixture to mock the _prerequisites private method in te GitHub action."""

    with patch.object(CustomModelsAction, "_prerequisites"):
        yield


@pytest.fixture
def mock_github_env_variables():
    """A fixture to mock GitHub environment variables."""

    default_env_vars = {"GITHUB_EVENT_NAME": "push", "GITHUB_BASE_REF": "HEAD~1"}
    with patch.dict(os.environ, default_env_vars):
        yield


@pytest.fixture
def mock_fetch_models_from_datarobot():
    """A fixture to patch the _fetch_models_from_datarobot private method."""

    with patch.object(ModelController, "fetch_models_from_datarobot"):
        yield


@pytest.fixture
def mock_model_version_exists():
    """A fixture to patch the _model_version_exists private method."""

    with patch.object(ModelController, "_model_version_exists", return_value=True):
        yield


@pytest.fixture
def mock_handle_deleted_models():
    """A fixture to patch the _handle_deleted_models private method."""

    with patch.object(ModelController, "handle_deleted_models", return_value=True):
        yield


@pytest.fixture(name="git_repo")
def fixture_git_repo(workspace_path):
    """A fixture to initialize a Git repository in a given root directory."""

    repo = Repo.init(workspace_path)
    repo.config_writer().set_value("user", "name", "test-user").release()
    repo.config_writer().set_value("user", "email", "test@company.com").release()
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    type(repo.git).GIT_PYTHON_TRACE = "full"
    return repo


@pytest.fixture(name="init_repo_for_root_path_factory")
def fixture_init_repo_for_root_path_factory(workspace_path, git_repo):
    """A fixture to commit all changes in a given repository"""

    def _inner():
        os.chdir(workspace_path)
        git_repo.git.add("--all")
        git_repo.git.commit("-m", "'Initial commit'", "--no-verify")
        return workspace_path

    return _inner


@pytest.fixture(name="init_repo_with_models_factory")
def fixture_init_repo_with_models_factory(
    workspace_path, models_factory, init_repo_for_root_path_factory
):
    """
    A fixture to create models in a given repository, commit the changes and return the
    repository file path.
    """

    def _inner(
        num_models=2,
        is_multi=False,
        with_include_glob=True,
        with_exclude_glob=True,
        include_main_prog=True,
    ):
        models_factory(
            num_models,
            is_multi,
            with_include_glob,
            with_exclude_glob,
            include_main_prog,
        )
        init_repo_for_root_path_factory()
        return workspace_path

    return _inner


@pytest.fixture(name="mock_full_custom_model_checks")
def fixture_mock_full_custom_model_checks():
    """A fixture to get a full custom model test checks."""

    return {
        ModelSchema.NULL_VALUE_IMPUTATION_KEY: {
            ModelSchema.CHECK_ENABLED_KEY: True,
            ModelSchema.BLOCK_DEPLOYMENT_IF_FAILS_KEY: True,
        },
        ModelSchema.SIDE_EFFECTS_KEY: {
            ModelSchema.CHECK_ENABLED_KEY: True,
            ModelSchema.BLOCK_DEPLOYMENT_IF_FAILS_KEY: True,
        },
        ModelSchema.PREDICTION_VERIFICATION_KEY: {
            ModelSchema.CHECK_ENABLED_KEY: True,
            ModelSchema.BLOCK_DEPLOYMENT_IF_FAILS_KEY: False,
            ModelSchema.OUTPUT_DATASET_ID_KEY: "627791f5562155d63f367b05",
            ModelSchema.PREDICTIONS_COLUMN: "Grade 2014",
            ModelSchema.MATCH_THRESHOLD_KEY: 0.9,
            ModelSchema.PASSING_MATCH_RATE_KEY: 85,
        },
        ModelSchema.PERFORMANCE_KEY: {
            ModelSchema.CHECK_ENABLED_KEY: True,
            ModelSchema.BLOCK_DEPLOYMENT_IF_FAILS_KEY: False,
            ModelSchema.MAXIMUM_RESPONSE_TIME_KEY: 50,
            ModelSchema.MAXIMUM_EXECUTION_TIME: 100,
            ModelSchema.NUMBER_OF_PARALLEL_USERS_KEY: 3,
        },
        ModelSchema.STABILITY_KEY: {
            ModelSchema.CHECK_ENABLED_KEY: True,
            ModelSchema.BLOCK_DEPLOYMENT_IF_FAILS_KEY: True,
            ModelSchema.TOTAL_PREDICTION_REQUESTS_KEY: 50,
            ModelSchema.PASSING_RATE_KEY: 95,
            ModelSchema.NUMBER_OF_PARALLEL_USERS_KEY: 1,
            ModelSchema.MINIMUM_PAYLOAD_SIZE_KEY: 100,
            ModelSchema.MAXIMUM_PAYLOAD_SIZE_KEY: 1000,
        },
    }


@pytest.fixture
def mock_full_binary_model_schema(mock_full_custom_model_checks):
    """A fixture to generate a full Binary model schema."""

    return {
        ModelSchema.MODEL_ID_KEY: "abc123",
        ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_BINARY,
        ModelSchema.SETTINGS_SECTION_KEY: {
            ModelSchema.NAME_KEY: "Awesome Model",
            ModelSchema.DESCRIPTION_KEY: "My awesome model",
            ModelSchema.TARGET_NAME_KEY: "target_column",
            ModelSchema.POSITIVE_CLASS_LABEL_KEY: "1",
            ModelSchema.NEGATIVE_CLASS_LABEL_KEY: "0",
            ModelSchema.LANGUAGE_KEY: "Python",
            ModelSchema.TRAINING_DATASET_ID_KEY: "627790ba56215587b3021632",
            ModelSchema.HOLDOUT_DATASET_ID_KEY: "627790ca5621558b55c78d78",
        },
        ModelSchema.VERSION_KEY: {
            ModelSchema.MODEL_ENV_ID_KEY: "627790db5621558eedc4c7fa",
            ModelSchema.INCLUDE_GLOB_KEY: ["./"],
            ModelSchema.EXCLUDE_GLOB_KEY: ["README.md", "out/"],
            ModelSchema.MEMORY_KEY: "100Mi",
            ModelSchema.REPLICAS_KEY: 3,
            ModelSchema.EGRESS_NETWORK_POLICY_KEY: ModelSchema.EGRESS_NETWORK_POLICY_PUBLIC,
        },
        ModelSchema.TEST_KEY: {
            ModelSchema.TEST_DATA_ID_KEY: "62779143562155aa34a3d65b",
            ModelSchema.TEST_SKIP_KEY: False,
            ModelSchema.MEMORY_KEY: "100Mi",
            ModelSchema.CHECKS_KEY: mock_full_custom_model_checks,
        },
    }


@pytest.fixture
def paginated_url_factory(webserver):
    """A fixture to emulate a paginated web page URL."""

    def _inner(base_url, page=0):
        suffix = "" if page == 0 else f"/page-{page}/"
        return f"{webserver}/api/v2/{base_url}{suffix}"

    return _inner


# pylint: disable=protected-access
def validate_metrics(github_output, entity_label, controller):
    """A method to validate models/deployments metrics."""

    controller._metrics.save()
    with open(github_output, "r", encoding="utf-8") as file:
        github_output_content = file.read()

    for metric_label in Metrics.metric_labels(entity_label):
        assert re.search(f"^{metric_label}=0$", github_output_content, re.M)

    desired_value = 5
    for metric in Metrics._get_metrics(entity_label):
        current_metric_value = getattr(controller._metrics, metric.name)
        desired_metric_value = Metric(current_metric_value.label, desired_value)
        setattr(controller._metrics, metric.name, desired_metric_value)

    controller._metrics.save()
    with open(github_output, "r", encoding="utf-8") as file:
        github_output_content = file.read()

    for metric_label in Metrics.metric_labels(entity_label):
        assert re.search(f"^{metric_label}={desired_value}$", github_output_content, re.M)

    assert not re.search(f"^.*{constants.Label.DEPLOYMENTS}.*$", github_output_content, flags=re.M)


@contextlib.contextmanager
def set_namespace(namespace):
    """
    A helper to set namespace and clean it up.

    Parameters
    ----------
    namespace : str
        A non-empty namespace name.
    """

    origin_namespace = Namespace.namespace()
    Namespace.uninit()
    try:
        Namespace.init(namespace)
        yield
    finally:
        Namespace.uninit()
        Namespace.init(origin_namespace)


def validate_namespaced_user_provided_id(info_bases, namespace):
    """
    A helper method to validate user provided IDs in a given namespace.

    Parameters
    ----------
    info_bases :  list[InfoBase]
        A list of entities, which can be one of ModelInfo or DeploymentInfo.
    namespace : str or None
        A valid namespace name or None.
    """

    for info in info_bases:
        assert Namespace.is_in_namespace(info.user_provided_id)
        if namespace:
            assert info.user_provided_id.startswith(f"{namespace}/")


@pytest.fixture(autouse=True)
def mock_git_model_version():
    """A fixture to mock the GitModelVersion class in the model controller."""

    with patch("model_controller.GitModelVersion"):
        yield
