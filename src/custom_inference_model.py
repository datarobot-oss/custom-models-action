from abc import ABC
from abc import abstractmethod
from enum import Enum
from glob import glob
from pathlib import Path
import logging
import os
import re
import yaml

from common.data_types import DataRobotModel
from common.data_types import FileInfo
from common.exceptions import ModelMainEntryPointNotFound
from common.exceptions import SharedAndLocalPathCollision
from common.exceptions import UnexpectedResult
from common.git_tool import GitTool
from dr_client import DrClient
from schema_validator import ModelSchema

logger = logging.getLogger()


class ModelInfo:
    def __init__(self, yaml_filepath, model_path, metadata):
        self._yaml_filepath = Path(yaml_filepath)
        self._model_path = Path(model_path)
        self._metadata = metadata
        self._model_file_paths = []
        self.should_upload_all_files = False
        self.changed_or_new_files = []
        self.deleted_file_ids = []

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
    def git_model_id(self):
        return self.metadata[ModelSchema.MODEL_ID_KEY]

    @property
    def model_file_paths(self):
        return self._model_file_paths

    @property
    def is_binary(self):
        return ModelSchema.is_binary(self.metadata)

    @property
    def is_regression(self):
        return ModelSchema.is_regression(self.metadata)

    @property
    def is_unstructured(self):
        return ModelSchema.is_unstructured(self.metadata)

    @property
    def is_multiclass(self):
        return ModelSchema.is_multiclass(self.metadata)

    def main_program_filepath(self):
        for p in self.model_file_paths:
            if p.name == "custom.py":
                return p
        return None

    def main_program_exists(self):
        return self.main_program_filepath() is not None

    def set_paths(self, paths):
        self._model_file_paths = [Path(p) for p in paths]

    @property
    def is_affected_by_commit(self):
        return (
            self.should_upload_all_files
            or len(self.changed_or_new_files) > 0
            or len(self.deleted_file_ids) > 0
        )

    @property
    def should_run_test(self):
        return ModelSchema.TEST_KEY in self.metadata and not ModelSchema.get_value(
            self.metadata, ModelSchema.TEST_KEY, ModelSchema.TEST_SKIP_KEY
        )


class CustomInferenceModelBase(ABC):
    def __init__(self, options):
        self._options = options
        self._repo = GitTool(self.options.root_dir)
        logger.info(f"GITHUB_EVENT_NAME: {self.event_name}")
        logger.info(f"GITHUB_SHA: {self.github_sha}")

    @property
    def options(self):
        return self._options

    @property
    def event_name(self):
        return os.environ.get("GITHUB_EVENT_NAME")

    @property
    def github_sha(self):
        return os.environ.get("GITHUB_SHA")

    @abstractmethod
    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models
        """
        pass


class CustomInferenceModel(CustomInferenceModelBase):
    class RelativeTo(Enum):
        ROOT = 1
        MODEL = 2

    def __init__(self, options):
        super().__init__(options)
        os.environ["GIT_PYTHON_TRACE"] = "full"
        self._models_info = []
        self._datarobot_models = {}
        self._dr_client = DrClient(self.options.webserver, self.options.api_token)

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
        self._fetch_models_from_datarobot()
        self._lookup_affected_models_by_the_current_action()
        self._apply_datarobot_actions_for_affected_models()

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
        supported_events = ["push", "pull_request"]
        if self.event_name not in supported_events:
            logger.warning(
                "Skip custom inference model action. It is expected to be executed only "
                f"on {supported_events} events. Current event: {self.event_name}."
            )
            return False

        base_ref = os.environ.get("GITHUB_BASE_REF")
        logger.info(f"GITHUB_BASE_REF: {base_ref}.")
        if self.event_name == "pull_request" and base_ref != self.options.branch:
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
                if ModelSchema.is_multi_models_schema(yaml_content):
                    transformed = ModelSchema.validate_and_transform_multi(yaml_content)
                    for model_entry in transformed[ModelSchema.MULTI_MODELS_KEY]:
                        model_path = self._to_absolute(
                            model_entry[ModelSchema.MODEL_ENTRY_PATH_KEY],
                            Path(yaml_path).parent,
                        )
                        model_metadata = model_entry[ModelSchema.MODEL_ENTRY_META_KEY]
                        model_info = ModelInfo(yaml_path, model_path, model_metadata)
                        self._models_info.append(model_info)
                elif ModelSchema.is_single_model_schema(yaml_content):
                    transformed = ModelSchema.validate_and_transform_single(yaml_content)
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
        # Handle this kind of paths: /a/./b/ ==> /a/b, /a//b ==> /a/b
        re_p1 = re.compile(r"/\./|//")
        # Handle this kind of path: ./a/b ==> a/b
        re_p2 = re.compile(r"^\./")
        paths = [re_p1.sub("/", p) for p in paths]
        return set([re_p2.sub("", p) for p in paths])

    def _validate_model_integrity(self, model_info):
        if not model_info.main_program_exists():
            raise ModelMainEntryPointNotFound(
                f"Model (Id: {model_info.git_model_id}) main entry point "
                f"not found (custom.py).\n"
                f"Existing files: {model_info.model_file_paths}"
            )

        self._validate_collision_between_local_and_shared(model_info)

    def _validate_collision_between_local_and_shared(self, model_info):
        model_file_paths = model_info.model_file_paths

        relative_paths = {self.RelativeTo.MODEL: set([]), self.RelativeTo.ROOT: set([])}
        for path in model_file_paths:
            relative_to, relative_path = self._get_relative_path(path, model_info)
            if not relative_to:
                raise UnexpectedResult(f"The path '{path}' is outside the repo.")
            if str(relative_path) != ".":
                relative_paths[relative_to].add(relative_path)

        collisions = relative_paths[self.RelativeTo.MODEL] & relative_paths[self.RelativeTo.ROOT]
        if collisions:
            raise SharedAndLocalPathCollision(
                f"Invalid file tree. Shared file(s)/package(s) collide with local model's "
                f"file(s)/package(s). Collisions: {collisions}."
            )

    def _get_relative_path(self, path, model_info):
        def _extract_path(p, root):
            relative_path = p.relative_to(root)
            extracted_path = relative_path.parts[0] if relative_path.parts else relative_path
            return extracted_path

        if self._is_relative_to(path, model_info.model_path):
            return self.RelativeTo.MODEL, _extract_path(path, model_info.model_path)
        elif self._is_relative_to(path, self.options.root_dir):
            return self.RelativeTo.ROOT, _extract_path(path, self.options.root_dir)
        else:
            return None, None

    @staticmethod
    def _is_relative_to(a_path, b_path):
        try:
            a_path.relative_to(b_path)
            return True
        except ValueError:
            return False

    def _fetch_models_from_datarobot(self):
        logger.info("Fetching models from DataRobot ...")
        custom_inference_models = self._dr_client.fetch_custom_models()
        for custom_model in custom_inference_models:
            git_model_id = custom_model.get("gitModelId")
            if git_model_id:
                model_versions = self._dr_client.fetch_custom_model_versions(
                    custom_model["id"], json={"limit": 1}
                )
                latest_version = model_versions[0] if model_versions else None
                if not latest_version:
                    logger.warning(
                        "Model exists without a version! git_model_id: "
                        f"{git_model_id}, custom_model_id: {custom_model['id']}"
                    )
                self._datarobot_models[git_model_id] = DataRobotModel(custom_model, latest_version)

    def _lookup_affected_models_by_the_current_action(self):
        logger.info("Lookup affected models by the current commit ...")

        for model_info in self.models_info:
            model_info.changed_or_new_files = []
            model_info.deleted_file_ids = []
            model_info.should_upload_all_files = self._should_upload_all_files(model_info)

        ancestor_ref = (
            "pullRequestCommitSha" if self.event_name == "pull_request" else "mainBranchCommitSha"
        )
        self._lookup_affected_models(ancestor_ref)

    def _should_upload_all_files(self, model_info):
        return (
            not self._model_version_exists(model_info)
            or self._is_dirty(model_info)
            or not self._valid_ancestor(model_info)
        )

    def _model_version_exists(self, model_info):
        return (
            model_info.git_model_id in self._datarobot_models
            and self._datarobot_models[model_info.git_model_id].latest_version
        )

    @staticmethod
    def _is_dirty(model_info):
        # TODO: Add support for 'dirty' in DataRobot for any action that was done by a non GitHub
        #       action client
        logger.warning(
            f"Add support to check 'dirty' marker for custom model version. "
            f"git_model_id: {model_info.git_model_id}"
        )
        return False

    def _valid_ancestor(self, model_info):
        ancestor_ref = (
            "pullRequestCommitSha" if self.event_name == "pull_requests" else "mainBranchCommitSha"
        )
        ancestor_sha = self._get_latest_provisioned_model_git_version(model_info)[ancestor_ref]
        if not ancestor_sha:
            # Either the model has never provisioned of the user created a version with a non
            # GitHub action client.
            return False

        # Users may have few local commits between remote pushes
        return self._repo.is_ancestor_of(ancestor_sha, self.github_sha)

    def _lookup_affected_models(self, ancestor_ref):
        # In a PR a merge commit is always the last commit, which we need to ignore.
        if logger.isEnabledFor(logging.DEBUG):
            self._repo.print_pretty_log()

        for model_info in self.models_info:
            if model_info.should_upload_all_files:
                continue
            from_commit_sha = self._get_latest_provisioned_model_git_version(model_info)[
                ancestor_ref
            ]
            changed_files, deleted_files = self._repo.find_changed_files(
                self.github_sha, from_commit_sha
            )
            self._handle_changed_or_new_files(model_info, changed_files)
            self._handle_deleted_files(model_info, deleted_files)

    @staticmethod
    def _handle_changed_or_new_files(model_info, changed_or_new_files):
        for changed_file in changed_or_new_files:
            if changed_file in model_info.model_file_paths:
                logger.info(
                    f"Changed/new file '{changed_file}' affects model "
                    f"'{model_info.model_path.name}'"
                )
                model_info.changed_or_new_files.append(changed_file)

    def _handle_deleted_files(self, model_info, deleted_files):
        for deleted_file in deleted_files:
            # Being stateless, check each deleted file against the stored custom model version
            # in DataRobot
            if model_info.git_model_id in self._datarobot_models:
                latest_version = self._datarobot_models[model_info.git_model_id].latest_version
                if latest_version:
                    _, relative_path = self._get_relative_path(deleted_file, model_info)
                    if not relative_path:
                        raise UnexpectedResult(f"The path '{deleted_file}' is outside the repo.")

                    model_info.deleted_file_ids.extend(
                        [
                            item["id"]
                            for item in latest_version["items"]
                            if relative_path == item["filePath"]
                        ]
                    )

    def _get_latest_provisioned_model_git_version(self, model_info):
        latest_version = self._datarobot_models[model_info.git_model_id].latest_version
        return latest_version["gitModelVersion"]

    def _apply_datarobot_actions_for_affected_models(self):
        logger.info("Apply DataRobot actions for affected models ...")
        for model_info in self._models_info:
            if model_info.is_affected_by_commit:
                logger.info(f"Model '{model_info.model_path}' is affected by commit.")

                custom_model_id = self._get_or_create_custom_model(model_info)
                version_id = self._create_custom_model_version(custom_model_id, model_info)
                if model_info.should_run_test:
                    self._test_custom_model_version(custom_model_id, version_id, model_info)

                logger.info(
                    "Custom inference model version was successfully created. "
                    f"git_model_id: {model_info.git_model_id}, model_id: {custom_model_id}, "
                    f"version_id: {version_id}"
                )

    def _get_or_create_custom_model(self, model_info):
        if model_info.git_model_id in self._datarobot_models:
            custom_model_id = self._datarobot_models[model_info.git_model_id].model["id"]
        else:
            custom_model_id = self._dr_client.create_custom_model(model_info)
            logger.info(f"Custom inference model was created: {custom_model_id}")
        return custom_model_id

    def _create_custom_model_version(self, custom_model_id, model_info):
        if model_info.should_upload_all_files:
            changed_files_info = self._get_relative_paths(model_info, model_info.model_file_paths)
        else:
            changed_files_info = self._get_relative_paths(
                model_info, model_info.changed_or_new_files
            )

        logger.info(
            "Create custom inference model version. git_model_id: "
            f" {model_info.git_model_id}, from_latest: {model_info.should_upload_all_files}"
        )

        if self.event_name == "pull_request":
            main_branch_commit_sha = self._repo.merge_base_commit_sha(
                self.options.branch, self.github_sha
            )
            pull_request_commit_sha = self._repo.feature_branch_top_commit_sha_of_a_merge_commit(
                self.github_sha
            )
        else:
            main_branch_commit_sha = self.github_sha
            pull_request_commit_sha = None

        return self._dr_client.create_custom_model_version(
            custom_model_id,
            model_info,
            main_branch_commit_sha,
            pull_request_commit_sha,
            changed_files_info,
            model_info.deleted_file_ids,
            from_latest=not model_info.should_upload_all_files,
        )

    def _get_relative_paths(self, model_info, paths, for_upload=True):
        file_references = []
        for path in paths:
            _, relative_path = self._get_relative_path(path, model_info)
            if str(relative_path) != ".":
                file_ref = FileInfo(path, relative_path) if for_upload else relative_path
                file_references.append(file_ref)
        return file_references

    def _test_custom_model_version(self, model_id, model_version_id, model_info):
        logger.info("Executing custom model test ...")
        self._dr_client.run_custom_model_version_testing(model_id, model_version_id, model_info)
