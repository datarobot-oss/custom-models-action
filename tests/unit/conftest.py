import logging
import os
import uuid
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
import yaml
from bson import ObjectId
from git import Repo

from custom_inference_model import CustomInferenceModel
from schema_validator import ModelSchema


@pytest.fixture
def repo_root_path():
    with TemporaryDirectory() as repo_tree:
        yield Path(repo_tree)


def _write_to_file(file_path, content):
    with open(file_path, "w") as f:
        f.write(content)


@pytest.fixture
def common_path(repo_root_path):
    return repo_root_path / "common"


@pytest.fixture
def common_filepath(common_path):
    return common_path / "common.py"


@pytest.fixture
def common_path_with_code(repo_root_path, common_path, common_filepath):
    os.makedirs(common_path)
    _write_to_file(common_filepath, "# common.py")
    _write_to_file(common_path / "util.py", "# Util.py")
    os.makedirs(common_path / "string")
    _write_to_file(common_path / "string" / "conv.py", "# conv.py")
    return common_path


@pytest.fixture
def common_package(common_dir):
    module_name = str(common_dir).replace("/", ".")
    return module_name


@pytest.fixture
def excluded_src_path(repo_root_path):
    excluded_path = repo_root_path / "excluded_path"
    os.makedirs(excluded_path)
    _write_to_file(excluded_path / "some_file.py", "# some_file.py")
    return excluded_path


@pytest.fixture
def single_model_factory(repo_root_path, common_path_with_code, excluded_src_path):
    def _inner(
        name,
        write_metadata=True,
        with_include_glob=True,
        with_exclude_glob=True,
        include_main_prog=True,
    ):
        model_path = repo_root_path / name
        os.makedirs(model_path)
        if include_main_prog:
            _write_to_file(model_path / "custom.py", "# custom.py")
        _write_to_file(model_path / "README.md", "# README")
        _write_to_file(model_path / "non-datarobot-yaml.yaml", '{"models": []}')
        os.makedirs(model_path / "score")
        _write_to_file(model_path / "score" / "score.py", "# score.py")

        single_model_metadata = {
            ModelSchema.MODEL_ID_KEY: str(uuid.uuid4()),
            ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION_KEY,
            ModelSchema.TARGET_NAME_KEY: "Grade 2014",
            ModelSchema.SETTINGS_KEY: {ModelSchema.NAME_KEY: "My Awesome Model"},
            ModelSchema.VERSION_KEY: {ModelSchema.MODEL_ENV_KEY: str(ObjectId())},
        }
        if with_include_glob:
            # noinspection PyTypeChecker
            single_model_metadata["version"]["include_glob_pattern"] = [
                "./**",
                f"/{common_path_with_code.relative_to(repo_root_path)}/**",
            ]
        if with_exclude_glob:
            # noinspection PyTypeChecker
            single_model_metadata["version"]["exclude_glob_pattern"] = ["./README.md"]

        if write_metadata:
            _write_to_file(model_path / "model.yaml", yaml.dump(single_model_metadata))

        return single_model_metadata

    yield _inner


@pytest.fixture
def models_factory(repo_root_path, common_path_with_code, single_model_factory):
    def _inner(
        num_models=2,
        is_multi=False,
        with_include_glob=True,
        with_exclude_glob=True,
        include_main_prog=True,
    ):
        multi_models_yaml_content = {ModelSchema.MULTI_MODELS_KEY: []} if is_multi else None
        for counter in range(num_models):
            model_name = f"model_{counter}"
            model_metadata = single_model_factory(
                model_name,
                write_metadata=not is_multi,
                with_include_glob=with_include_glob,
                with_exclude_glob=with_exclude_glob,
                include_main_prog=include_main_prog,
            )
            if is_multi:
                multi_models_yaml_content[ModelSchema.MULTI_MODELS_KEY].append(
                    {
                        ModelSchema.MODEL_ENTRY_PATH_KEY: f"./{model_name}",
                        ModelSchema.MODEL_ENTRY_META_KEY: model_metadata,
                    }
                )
        if is_multi:
            multi_models_content = yaml.dump(multi_models_yaml_content)
            _write_to_file(repo_root_path / "models.yaml", multi_models_content)

        return repo_root_path

    return _inner


@pytest.fixture
def single_model_file_paths(models_factory, repo_root_path):
    models_factory(1)
    model_path = repo_root_path / "model_0"
    return list(model_path.rglob("*.*"))


@pytest.fixture
def options(repo_root_path):
    return Namespace(
        webserver="www.dummy.com",
        api_token="abc123",
        root_dir=repo_root_path.absolute(),
        branch="master",
    )


@pytest.fixture
def mock_prerequisites():
    with patch.object(CustomInferenceModel, "_prerequisites"):
        yield


@pytest.fixture
def mock_github_env_variables():
    default_env_vars = {"GITHUB_EVENT_NAME": "push", "GITHUB_BASE_REF": "HEAD~1"}
    with patch.dict(os.environ, default_env_vars):
        yield


@pytest.fixture
def mock_fetch_models_from_datarobot():
    with patch.object(CustomInferenceModel, "_fetch_models_from_datarobot"):
        yield


@pytest.fixture
def mock_model_version_exists():
    with patch.object(CustomInferenceModel, "_model_version_exists", return_value=True):
        yield


@pytest.fixture
def git_repo(repo_root_path):
    repo = Repo.init(repo_root_path)
    repo.config_writer().set_value("user", "name", "test-user").release()
    repo.config_writer().set_value("user", "email", "test@company.com").release()
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)
    type(repo.git).GIT_PYTHON_TRACE = "full"
    return repo


@pytest.fixture
def init_repo_with_models_factory(git_repo, repo_root_path, models_factory):
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
        os.chdir(repo_root_path)
        git_repo.git.add("--all")
        git_repo.git.commit("-m", "'Initial commit'", "--no-verify")
        return repo_root_path

    return _inner


@pytest.fixture
def mock_full_custom_model_checks():
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
            ModelSchema.OUTPUT_DATASET_KEY: "627791f5562155d63f367b05",
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
    return {
        ModelSchema.MODEL_ID_KEY: "abc123",
        ModelSchema.DEPLOYMENT_ID_KEY: "edf456",
        ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_BINARY_KEY,
        ModelSchema.TARGET_NAME_KEY: "target_column",
        ModelSchema.POSITIVE_CLASS_LABEL_KEY: "1",
        ModelSchema.NEGATIVE_CLASS_LABEL_KEY: "0",
        ModelSchema.LANGUAGE_KEY: "Python",
        ModelSchema.SETTINGS_KEY: {
            ModelSchema.NAME_KEY: "Awesome Model",
            ModelSchema.DESCRIPTION_KEY: "My awesome model",
            ModelSchema.TRAINING_DATASET_KEY: "627790ba56215587b3021632",
            ModelSchema.HOLDOUT_DATASET_KEY: "627790ca5621558b55c78d78",
        },
        ModelSchema.VERSION_KEY: {
            ModelSchema.MODEL_ENV_KEY: "627790db5621558eedc4c7fa",
            ModelSchema.INCLUDE_GLOB_KEY: ["./"],
            ModelSchema.EXCLUDE_GLOB_KEY: ["README.md", "out/"],
            ModelSchema.MEMORY_KEY: "100Mi",
            ModelSchema.REPLICAS_KEY: 3,
        },
        ModelSchema.TEST_KEY: {
            ModelSchema.TEST_DATA_KEY: "62779143562155aa34a3d65b",
            ModelSchema.TEST_SKIP_KEY: False,
            ModelSchema.MEMORY_KEY: "100Mi",
            ModelSchema.CHECKS_KEY: mock_full_custom_model_checks,
        },
    }


@pytest.fixture
def paginated_url_factory(webserver):
    def _inner(base_url, page=0):
        suffix = "" if page == 0 else f"/page-{page}/"
        return f"{webserver}/api/v2/{base_url}{suffix}"

    return _inner
