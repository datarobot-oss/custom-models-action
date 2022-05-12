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
            "target_type": "Regression",
            "target_name": "Grade 2014",
            "version": {"model_environment": str(ObjectId())},
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
def options(repo_root_path):
    return Namespace(root_dir=repo_root_path.absolute(), branch="master")


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
