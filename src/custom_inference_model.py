import os
import re
from abc import ABC
from abc import abstractmethod
import logging
from glob import glob
from pathlib import Path

import yaml
from git import Repo

from exceptions import ModelMainEntryPointNotFound, SharedAndLocalPathCollision
from git_tool import GitTool
from schema_validator import ModelSchema

logger = logging.getLogger()


class ModelInfo:
    def __init__(self, yaml_filepath, model_path, metadata):
        self._yaml_filepath = Path(yaml_filepath)
        self._model_path = Path(model_path)
        self._metadata = metadata
        self._model_file_paths = []
        self.is_affected_by_commit = False

    @property
    def yaml_filepath(self):
        return self._yaml_filepath

    @property
    def model_path(self):
        return self._model_path

    @property
    def metadata(self):
        return self._metadata

    @property
    def model_file_paths(self):
        return self._model_file_paths

    def main_program_filepath(self):
        for p in self.model_file_paths:
            if p.name == "custom.py":
                return p
        return None

    def main_program_exists(self):
        return self.main_program_filepath() is not None

    def set_paths(self, paths):
        self._model_file_paths = [Path(p) for p in paths]


class CustomInferenceModelBase(ABC):
    def __init__(self, options):
        self._options = options
        self._repo = GitTool(self.options.root_dir)

    @property
    def options(self):
        return self._options

    @abstractmethod
    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models
        """
        pass


class CustomInferenceModel(CustomInferenceModelBase):
    def __init__(self, options):
        super().__init__(options)
        os.environ["GIT_PYTHON_TRACE"] = "full"
        self._models_info = []
        self._model_schema = ModelSchema()

    @property
    def models_info(self):
        return self._models_info

    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models

        This method implements the following logic:
        1. Scan and load DataRobot model metadata (yaml files)
        1. Go over all the models and per model collect the files/folders belong to the model
        2. Per changed file/folder in the commit, find all affected models
        3. Per affected model, create a new version in DataRobot and run tests.
        """

        if not self._prerequisites():
            return

        logger.info(f"Options: {self.options}")

        self._scan_and_load_datarobot_models_metadata()
        self._collect_datarobot_model_files()
        self._lookup_affected_models_by_the_current_action()
        self._fetch_models_from_datarobot()
        self._create_affected_model_versions_in_datarobot()

        print(
            """
            ::set-output name=new-model-created::True
            ::set-output name=model-deleted::False
            ::set-output name=new-model-version-created::True
            ::set-output name=test-result::The test passed with success.
            ::set-output name=returned-code::200.
            ::set-output name=message::Custom model created and tested with success.
            """
        )

    def _prerequisites(self):
        logger.info(f"GITHUB_SHA: {os.environ.get('GITHUB_SHA')}")

        github_event_name = os.environ.get("GITHUB_EVENT_NAME")
        logger.info(f"GITHUB_EVENT_NAME: {github_event_name}")

        supported_events = ["push", "pull_request"]
        if github_event_name not in supported_events:
            logger.warning(
                "Skip custom inference model action. It is expected to be executed only "
                f"on {supported_events} events. Current event: {github_event_name}."
            )
            return False

        base_ref = os.environ.get("GITHUB_BASE_REF")
        logger.info(f"GITHUB_BASE_REF: {base_ref}.")
        if github_event_name == "pull_request" and base_ref != self.options.branch:
            logger.info(
                "Skip custom inference model action. It is executed only when the referenced "
                f"branch is {self.options.branch}. Current ref branch: {base_ref}."
            )
            return False

        num_commits = self._repo.num_commits()
        if num_commits < 2:
            logger.warning(
                "Skip custom inference model action. The minimum number of commits should be 2. "
                f"Current number is {num_commits}"
            )
            return False
        return True

    def _scan_and_load_datarobot_models_metadata(self):
        yaml_files = glob(f"{self.options.root_dir}/**/*.yaml", recursive=True)
        yaml_files.extend(glob(f"{self.options.root_dir}/**/*.yml", recursive=True))
        for yaml_path in yaml_files:
            with open(yaml_path) as f:
                yaml_content = yaml.safe_load(f)
                if self._model_schema.is_multi_models_schema(yaml_content):
                    transformed = self._model_schema.validate_and_transform_multi(yaml_content)
                    for model_entry in transformed[self._model_schema.MULTI_MODELS_KEY]:
                        model_path = self._to_absolute(
                            model_entry[ModelSchema.MODEL_ENTRY_PATH_KEY],
                            Path(yaml_path).parent,
                        )
                        model_metadata = model_entry[ModelSchema.MODEL_ENTRY_META_KEY]
                        model_info = ModelInfo(yaml_path, model_path, model_metadata)
                        self._models_info.append(model_info)
                elif self._model_schema.is_single_model_schema(yaml_content):
                    transformed = self._model_schema.validate_and_transform_single(yaml_content)
                    yaml_path = Path(yaml_path)
                    model_info = ModelInfo(yaml_path, yaml_path.parent, transformed)
                    self._models_info.append(model_info)

    def _to_absolute(self, path, parent):
        match = re.match(r"^(/|\$ROOT/)", path)
        if match:
            path = path.replace(match[0], "", 1)
            path = f"{self.options.root_dir}/{path}"
        else:
            path = f"{parent}/{path}"
        return path

    def _collect_datarobot_model_files(self):
        for model_info in self.models_info:
            include_glob_patterns = model_info.metadata[ModelSchema.VERSION_KEY][
                ModelSchema.INCLUDE_GLOB_KEY
            ]
            included_paths = set([])
            if include_glob_patterns:
                for include_glob_pattern in include_glob_patterns:
                    include_glob_pattern = self._to_absolute(
                        include_glob_pattern, model_info.model_path
                    )
                    included_paths.update(glob(include_glob_pattern, recursive=True))
            else:
                included_paths.update(glob(f"{model_info.model_path}/**", recursive=True))

            excluded_paths = set([])
            exclude_glob_patterns = model_info.metadata[ModelSchema.VERSION_KEY][
                ModelSchema.EXCLUDE_GLOB_KEY
            ]
            for exclude_glob_pattern in exclude_glob_patterns:
                exclude_glob_pattern = self._to_absolute(
                    exclude_glob_pattern, model_info.model_path
                )
                # For excluded directories always assume recursive
                if Path(exclude_glob_pattern).is_dir():
                    exclude_glob_pattern += "/**"

                excluded_paths.update(glob(exclude_glob_pattern, recursive=True))

            self._set_filtered_model_paths(model_info, included_paths, excluded_paths)
            self._validate_model_integrity(model_info)
            logger.info(f"Model {model_info.model_path} detected and verified.")

    @classmethod
    def _set_filtered_model_paths(cls, model_info, included_paths, excluded_paths):
        included_paths = cls._normalize_paths(included_paths)
        excluded_paths = cls._normalize_paths(excluded_paths)
        final_model_paths = included_paths - excluded_paths
        model_info.set_paths(final_model_paths)

    @staticmethod
    def _normalize_paths(paths):
        re_p1 = re.compile(r"/\./|//")
        re_p2 = re.compile(r"^\./")
        paths = [re_p1.sub("/", p) for p in paths]
        return set([re_p2.sub("", p) for p in paths])

    def _validate_model_integrity(self, model_info):
        if not model_info.main_program_exists():
            raise ModelMainEntryPointNotFound(
                f"Model (Id: {model_info.metadata[ModelSchema.MODEL_ID_KEY]}) main entry point "
                f"not found (custom.py).\n"
                f"Existing files: {model_info.model_file_paths}"
            )

        self._validate_collision_between_local_and_shared(model_info)

    def _validate_collision_between_local_and_shared(self, model_info):
        model_path = model_info.model_path
        model_file_paths = model_info.model_file_paths

        local_path_top_levels = set([])
        shared_path_top_levels = set([])
        for path in model_file_paths:
            if self._is_relative_to(path, model_path):
                relative_path = path.relative_to(model_path)
                if str(relative_path) != ".":
                    local_path_top_levels.add(relative_path.parts[0])
            elif self._is_relative_to(path, self.options.root_dir):
                relative_path = path.relative_to(self.options.root_dir)
                if str(relative_path) != ".":
                    shared_path_top_levels.add(relative_path.parts[0])

        collisions = set(local_path_top_levels) & set(shared_path_top_levels)
        if collisions:
            raise SharedAndLocalPathCollision(
                f"Invalid file tree. Shared file(s)/package(s) collide with local model's "
                f"file(s)/package(s). Collisions: {collisions}."
            )

    @staticmethod
    def _is_relative_to(a_path, b_path):
        try:
            a_path.relative_to(b_path)
            return True
        except ValueError:
            return False

    def _lookup_affected_models_by_the_current_action(self):
        logger.info("Lookup affected models by the current commit ...")

        for model_info in self.models_info:
            model_info.is_affected_by_commit = False

        github_event_name = os.environ.get("GITHUB_EVENT_NAME")
        if github_event_name == "pull_request":
            self._lookup_affected_models_by_a_pull_request_action()
        elif github_event_name == "push":
            self._lookup_affected_models_by_a_push_action()

    def _lookup_affected_models_by_a_pull_request_action(self):
        # In a PR a merge commit is always the last commit, which we need to ignore.
        merge_commit_sha = os.environ.get("GITHUB_SHA")
        last_commit_sha = f"{merge_commit_sha}~1"
        changed_files = self._repo.find_changed_files(last_commit_sha)
        for changed_file in changed_files:
            for model_info in self.models_info:
                if changed_file in model_info.model_file_paths:
                    logger.info(
                        f"Changed file '{changed_file}' affects model "
                        f"'{model_info.model_path.name}'"
                    )
                    model_info.is_affected_by_commit = True

    def _lookup_affected_models_by_a_push_action(self):
        # When a PR is merged to the main branch, we try to find the relevant files that
        # were changed since the last provision to DataRobot.
        for model_info in self.models_info:
            to_commit_sha = os.environ.get("GITHUB_SHA")
            from_commit_sha = self._get_last_model_provisioned_git_sha(model_info)
            if from_commit_sha is None:
                # The assumption is that the model has never provisioned, so mark is as affected by
                # the given commit
                model_info.is_affected_by_commit = True
            else:
                changed_files = self._repo.find_changed_files(to_commit_sha, from_commit_sha)
                for changed_file in changed_files:
                    if changed_file in model_info.model_file_paths:
                        logger.info(
                            f"Changed file '{changed_file}' affects model "
                            f"'{model_info.model_path.name}'"
                        )
                        model_info.is_affected_by_commit = True

    def _get_last_model_provisioned_git_sha(self, model_info):
        # TODO: read the last provisioned git sha from DataRobot CustomTask entity
        return None

    def _find_changed_files(self, from_commit_sha, to_commit_sha):
        repo = Repo(self.options.root_dir)
        to_commit = repo.commit(to_commit_sha)
        diff = to_commit.diff(from_commit_sha)

        changed_files = []
        for git_index in diff:
            changed_files.append(self.options.root_dir / git_index.a_path)

        return changed_files

    def _fetch_models_from_datarobot(self):
        logger.info("Fetching models from DataRobot ...")

    def _create_affected_model_versions_in_datarobot(self):
        logger.info("Create affected model version in DataRobot ...")
