from abc import ABC
from abc import abstractmethod
from enum import Enum
from glob import glob
from pathlib import Path
import logging
import os
import re
from typing import Dict

import yaml

from common.data_types import DataRobotModel
from common.exceptions import DataRobotClientError
from common.exceptions import IllegalModelDeletion
from common.exceptions import ModelMainEntryPointNotFound
from common.exceptions import ModelMetadataAlreadyExists
from common.exceptions import PathOutsideTheRepository
from common.exceptions import SharedAndLocalPathCollision
from common.exceptions import UnexpectedResult
from common.git_tool import GitTool
from dr_client import DrClient
from schema_validator import ModelSchema

logger = logging.getLogger()


class ModelFilePath:
    class RelativeTo(Enum):
        MODEL = 1
        ROOT = 2

    def __init__(self, raw_file_path, model_root_dir, repo_root_dir):
        self._raw_file_path = raw_file_path
        self._filepath = Path(raw_file_path)
        # It is important to have an indication about the path origin and the relation to
        # the model, means whether the given path was originally under the model's root dir
        # or it is supposed to be copied into it. This will help us to detect collisions
        # between paths that exist under the model versus those that are supposed to be copied.
        path_under_model, relative_to = self.get_path_under_model(
            self._filepath, model_root_dir, repo_root_dir
        )
        self._under_model = path_under_model
        self._relative_to = relative_to

    @classmethod
    def get_path_under_model(cls, filepath, model_root_dir, repo_root_dir):
        try:
            path_under_model = cls._get_path_under_model_for_given_root(filepath, model_root_dir)
            relative_to = cls.RelativeTo.MODEL
        except ValueError:
            try:
                path_under_model = cls._get_path_under_model_for_given_root(filepath, repo_root_dir)
                relative_to = cls.RelativeTo.ROOT
            except ValueError:
                raise PathOutsideTheRepository(
                    f"Model file path is outside the repository: {filepath}"
                )
        return path_under_model, relative_to

    @staticmethod
    def _get_path_under_model_for_given_root(filepath, root):
        relative_path = filepath.relative_to(root)
        return str(relative_path).replace("../", "")  # Will be copied under the model

    def __str__(self):
        return self.under_model

    @property
    def filepath(self):
        return self._filepath

    @property
    def name(self):
        return self.filepath.name

    @property
    def resolved(self):
        return self.filepath.resolve()

    @property
    def under_model(self):
        return self._under_model

    @property
    def relative_to(self):
        return self._relative_to


class ModelInfo:
    _model_file_paths: Dict[Path, ModelFilePath]

    def __init__(self, yaml_filepath, model_path, metadata):
        self._yaml_filepath = Path(yaml_filepath)
        self._model_path = Path(model_path)
        self._metadata = metadata
        self._model_file_paths = {}
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
        try:
            return next(p for _, p in self.model_file_paths.items() if p.name == "custom.py")
        except StopIteration:
            return None

    def main_program_exists(self):
        return self.main_program_filepath() is not None

    def set_model_paths(self, paths, repo_root_path):
        self._model_file_paths = {}
        for p in paths:
            model_filepath = ModelFilePath(p, self.model_path, repo_root_path)
            self._model_file_paths[model_filepath.resolved] = model_filepath

    def paths_under_model_by_relative(self, relative_to):
        return set(
            [
                p.under_model
                for _, p in self.model_file_paths.items()
                if p.relative_to == relative_to
            ]
        )

    @property
    def is_affected_by_commit(self):
        return (
            self.should_upload_all_files
            or len(self.changed_or_new_files) > 0
            or len(self.deleted_file_ids) > 0
        )

    @property
    def should_run_test(self):
        return ModelSchema.TEST_KEY in self.metadata and not self.get_value(
            ModelSchema.TEST_KEY, ModelSchema.TEST_SKIP_KEY
        )

    def get_value(self, *args):
        return ModelSchema.get_value(self.metadata, *args)


class CustomInferenceModelBase(ABC):
    _models_info: Dict[str, ModelInfo]

    def __init__(self, options):
        self._options = options
        self._repo = GitTool(self.options.root_dir)
        self._models_info = {}
        self._datarobot_models = {}
        self._datarobot_models_by_id = {}
        self._dr_client = DrClient(
            self.options.webserver,
            self.options.api_token,
            verify_cert=not self.options.skip_cert_verification,
        )
        self._total_affected = 0
        self._total_created = 0
        self._total_deleted = 0
        logger.info(f"GITHUB_EVENT_NAME: {self.event_name}")
        logger.info(f"GITHUB_SHA: {self.github_sha}")
        logger.info(f"GITHUB_REPOSITORY: {self.github_repository}")

    @property
    def options(self):
        return self._options

    @property
    def event_name(self):
        return os.environ.get("GITHUB_EVENT_NAME")

    @property
    def is_pull_request(self):
        return self.event_name == "pull_request"

    @property
    def is_push(self):
        return self.event_name == "push"

    @property
    def ancestor_attribute_ref(self):
        return "pullRequestCommitSha" if self.is_pull_request else "mainBranchCommitSha"

    @property
    def github_sha(self):
        return os.environ.get("GITHUB_SHA")

    @property
    def github_repository(self):
        return os.environ.get("GITHUB_REPOSITORY")

    @property
    def models_info(self):
        return self._models_info

    @property
    def datarobot_models(self):
        return self._datarobot_models

    def datarobot_model_by_id(self, model_id):
        return self._datarobot_models_by_id.get(model_id)

    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models
        """
        try:
            if not self._prerequisites():
                return

            self._scan_and_load_models_metadata()
            self._run()
        finally:
            print(
                f"""
                ::set-output name=total-affected-{self._label()}::{self._total_affected}
                ::set-output name=total-created-{self._label()}::{self._total_created}
                ::set-output name=total-deleted-{self._label()}::{self._total_deleted}
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
        if self.is_pull_request and base_ref != self.options.branch:
            logger.info(
                "Skip custom inference model action. It is executed only when the referenced "
                f"branch is {self.options.branch}. Current ref branch: {base_ref}."
            )
            return False

        # NOTE: in the case of functional tests, the number of remotes is zero and still it's valid.
        if self._repo.num_remotes() > 1:
            logger.warning(
                "Skip custom inference model action, because the given repository has more than "
                "one remote configured."
            )
            return False

        if self.is_pull_request:
            # For pull request we assume a merge branch, which contains at least 2 commits
            num_commits = self._repo.num_commits()
            if num_commits < 2:
                logger.warning(
                    "Skip custom inference model action. The minimum number of commits "
                    f"should be 2. Current number is {num_commits}."
                )
                return False
        return True

    def _scan_and_load_models_metadata(self):
        logger.info("Scanning and loading DataRobot model files ...")
        for yaml_path, yaml_content in self._next_yaml_content_in_repo():
            if ModelSchema.is_multi_models_schema(yaml_content):
                transformed = ModelSchema.validate_and_transform_multi(yaml_content)
                for model_entry in transformed[ModelSchema.MULTI_MODELS_KEY]:
                    model_path = self._to_absolute(
                        model_entry[ModelSchema.MODEL_ENTRY_PATH_KEY],
                        Path(yaml_path).parent,
                    )
                    model_metadata = model_entry[ModelSchema.MODEL_ENTRY_META_KEY]
                    model_info = ModelInfo(yaml_path, model_path, model_metadata)
                    self._add_new_model_info(model_info)
            elif ModelSchema.is_single_model_schema(yaml_content):
                transformed = ModelSchema.validate_and_transform_single(yaml_content)
                yaml_path = Path(yaml_path)
                model_info = ModelInfo(yaml_path, yaml_path.parent, transformed)
                self._add_new_model_info(model_info)

    def _next_yaml_content_in_repo(self):
        yaml_files = glob(f"{self.options.root_dir}/**/*.yaml", recursive=True)
        yaml_files.extend(glob(f"{self.options.root_dir}/**/*.yml", recursive=True))
        for yaml_path in yaml_files:
            with open(yaml_path) as f:
                yield yaml_path, yaml.safe_load(f)

    def _to_absolute(self, path, parent):
        match = re.match(r"^(/|\$ROOT/)", path)
        if match:
            path = path.replace(match[0], "", 1)
            path = f"{self.options.root_dir}/{path}"
        else:
            path = f"{parent}/{path}"
        return path

    def _add_new_model_info(self, model_info):
        if model_info.git_model_id in self.models_info:
            raise ModelMetadataAlreadyExists(
                f"Model {model_info.git_model_id} already exists. "
                f"New model yaml path: {model_info.yaml_filepath}. "
            )

        logger.info(
            f"Adding new model metadata. Git model ID: {model_info.git_model_id}. "
            f"Model metadata yaml path: {model_info.yaml_filepath}."
        )
        self.models_info[model_info.git_model_id] = model_info

    def _fetch_models_from_datarobot(self):
        logger.info("Fetching models from DataRobot ...")
        custom_inference_models = self._dr_client.fetch_custom_models()
        for custom_model in custom_inference_models:
            git_model_id = custom_model.get("gitModelId")
            if git_model_id:
                datarobot_model_id = custom_model["id"]
                model_versions = self._dr_client.fetch_custom_model_versions(
                    datarobot_model_id, json={"limit": 1}
                )
                latest_version = model_versions[0] if model_versions else None
                if not latest_version:
                    logger.warning(
                        "Model exists without a version! git_model_id: "
                        f"{git_model_id}, custom_model_id: {datarobot_model_id}"
                    )
                datarobot_model = DataRobotModel(custom_model, latest_version)
                self.datarobot_models[git_model_id] = datarobot_model
                self._datarobot_models_by_id[datarobot_model_id] = datarobot_model

    def _get_latest_model_version_git_commit_ancestor(self, model_info):
        latest_version = self.datarobot_models[model_info.git_model_id].latest_version
        git_model_version = latest_version.get("gitModelVersion")
        if not git_model_version:
            # Either the model has never provisioned of the user created a version with a non
            # GitHub action client.
            return False

        return git_model_version[self.ancestor_attribute_ref]

    @abstractmethod
    def _label(self):
        pass

    @abstractmethod
    def _run(self):
        pass


class CustomInferenceModel(CustomInferenceModelBase):
    def __init__(self, options):
        super().__init__(options)

    def _label(self):
        return "models"

    def _run(self):
        """
        Executes the GitHub action logic to manage custom inference models

        This method implements the following logic:
        1. Scan and load DataRobot model metadata (yaml files)
        1. Go over all the models and per model collect the files/folders belong to the model
        2. Per changed file/folder in the commit, find all affected models
        3. Per affected model, create a new version in DataRobot and run tests.
        """

        try:
            self._collect_datarobot_model_files()
            self._fetch_models_from_datarobot()
            self._lookup_affected_models_by_the_current_action()
            self._apply_datarobot_actions_for_affected_models()
        finally:
            print(
                f"""
                ::set-output name=total-affected-models::{self._total_affected}
                ::set-output name=total-created-models::{self._total_created}
                ::set-output name=total-deleted-models::{self._total_deleted}
                """
            )

    def _collect_datarobot_model_files(self):
        logger.info("Collecting DataRobot model files ...")
        for _, model_info in self.models_info.items():
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

            self._set_filtered_model_paths(
                model_info, included_paths, excluded_paths, self.options.root_dir
            )
            self._validate_model_integrity(model_info)
            logger.info(f"Model {model_info.model_path} detected and verified.")

    @classmethod
    def _set_filtered_model_paths(cls, model_info, included_paths, excluded_paths, repo_root_dir):
        final_model_paths = []
        included_paths = cls._remove_undesired_sub_paths(included_paths)
        if excluded_paths:
            excluded_paths = cls._remove_undesired_sub_paths(excluded_paths)
            excluded_normpaths = [os.path.normpath(p) for p in excluded_paths]
        else:
            excluded_normpaths = []
        model_root_dir = str(model_info.model_path.absolute())
        for included_path in included_paths:
            included_normpath = os.path.normpath(included_path)
            if included_normpath in excluded_normpaths:
                continue
            if included_normpath == model_root_dir:
                continue
            if os.path.isdir(included_normpath):
                continue
            final_model_paths.append(included_path)

        model_info.set_model_paths(final_model_paths, repo_root_dir)

    @staticmethod
    def _remove_undesired_sub_paths(paths):
        # NOTE: we would like to keep relative paths without resolving them.
        # Handle this kind of paths: /a/./b/ ==> /a/b, /a//b ==> /a/b
        re_p1 = re.compile(r"/\./|//")
        # Handle this kind of path: ./a/b ==> a/b
        re_p2 = re.compile(r"^\./")
        paths = [re_p1.sub("/", p) for p in paths]
        return [re_p2.sub("", p) for p in paths]

    def _validate_model_integrity(self, model_info):
        if not model_info.main_program_exists():
            raise ModelMainEntryPointNotFound(
                f"Model (Id: {model_info.git_model_id}) main entry point "
                f"not found (custom.py).\n"
                f"Existing files: {model_info.model_file_paths}"
            )

        self._validate_collision_between_local_and_shared(model_info)

    @staticmethod
    def _validate_collision_between_local_and_shared(model_info):
        paths_relative_to_model = model_info.paths_under_model_by_relative(
            ModelFilePath.RelativeTo.MODEL
        )
        paths_relative_to_root = model_info.paths_under_model_by_relative(
            ModelFilePath.RelativeTo.ROOT
        )
        collisions = paths_relative_to_model & paths_relative_to_root
        if collisions:
            raise SharedAndLocalPathCollision(
                f"Invalid file tree. Shared file(s)/package(s) collide with local model's "
                f"file(s)/package(s). Collisions: {collisions}."
            )

    def _lookup_affected_models_by_the_current_action(self):
        logger.info("Lookup affected models by the current commit ...")

        for _, model_info in self.models_info.items():
            model_info.should_upload_all_files = self._should_upload_all_files(model_info)

        self._lookup_affected_models()

    def _should_upload_all_files(self, model_info):
        return (
            not self._model_version_exists(model_info)
            or self._is_dirty(model_info)
            or not self._valid_ancestor(model_info)
        )

    def _model_version_exists(self, model_info):
        return (
            model_info.git_model_id in self.datarobot_models
            and self.datarobot_models[model_info.git_model_id].latest_version
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
        ancestor_sha = self._get_latest_model_version_git_commit_ancestor(model_info)
        if not ancestor_sha:
            return False

        # Users may have few local commits between remote pushes
        is_ancestor = self._repo.is_ancestor_of(ancestor_sha, self.github_sha)
        logger.debug(
            f"Is the latest model version's git commit sha ({ancestor_sha}) "
            f"ancestor of the current commit ({self.github_sha})? Answer: {is_ancestor}"
        )
        return is_ancestor

    def _lookup_affected_models(self):
        # In a PR a merge commit is always the last commit, which we need to ignore.
        if logger.isEnabledFor(logging.DEBUG):
            self._repo.print_pretty_log()

        for _, model_info in self.models_info.items():
            model_info.changed_or_new_files = []
            model_info.deleted_file_ids = []

            if model_info.should_upload_all_files:
                continue

            from_commit_sha = self._get_latest_model_version_git_commit_ancestor(model_info)
            if not from_commit_sha:
                raise UnexpectedResult(
                    f"Unexpected None ancestor commit sha, "
                    f"model_git_id: {model_info.git_model_id}"
                )
            changed_files, deleted_files = self._repo.find_changed_files(
                self.github_sha, from_commit_sha
            )
            logger.debug(f"Changed files: {changed_files}. Deleted files: {deleted_files}")
            self._handle_changed_or_new_files(model_info, changed_files)
            self._handle_deleted_files(model_info, deleted_files)

    @staticmethod
    def _handle_changed_or_new_files(model_info, changed_or_new_files):
        for changed_file in changed_or_new_files:
            changed_model_filepath = model_info.model_file_paths.get(changed_file)
            if changed_model_filepath:
                logger.info(
                    f"Changed/new file '{changed_file}' affects model "
                    f"'{model_info.model_path.name}'"
                )
                model_info.changed_or_new_files.append(changed_model_filepath)

    def _handle_deleted_files(self, model_info, deleted_files):
        for deleted_file in deleted_files:
            # Being stateless, check each deleted file against the stored custom model version
            # in DataRobot
            if model_info.git_model_id in self.datarobot_models:
                latest_version = self.datarobot_models[model_info.git_model_id].latest_version
                if latest_version:
                    model_root_dir = model_info.model_path
                    repo_root_dir = self.options.root_dir
                    path_under_model, relative_to = ModelFilePath.get_path_under_model(
                        deleted_file, model_root_dir, repo_root_dir
                    )
                    if not path_under_model:
                        raise UnexpectedResult(f"The path '{deleted_file}' is outside the repo.")

                    model_info.deleted_file_ids.extend(
                        [
                            item["id"]
                            for item in latest_version["items"]
                            if path_under_model == item["filePath"]
                        ]
                    )

    def _apply_datarobot_actions_for_affected_models(self):
        logger.info("Apply DataRobot actions for affected models ...")
        self._handle_model_changes_or_creation()
        self._handle_deleted_models()

    def _handle_model_changes_or_creation(self):
        for git_model_id, model_info in self.models_info.items():
            if model_info.is_affected_by_commit:
                logger.info(f"Model '{model_info.model_path}' is affected by commit.")

                custom_model_id, already_existed = self._get_or_create_custom_model(model_info)
                if not already_existed:
                    self._total_created += 1
                version_id = self._create_custom_model_version(custom_model_id, model_info)
                if model_info.should_run_test:
                    self._test_custom_model_version(custom_model_id, version_id, model_info)

                self._total_affected += 1
                logger.info(
                    "Custom inference model version was successfully created. "
                    f"git_model_id: {git_model_id}, model_id: {custom_model_id}, "
                    f"version_id: {version_id}"
                )

    def _get_or_create_custom_model(self, model_info):
        already_exists = model_info.git_model_id in self.datarobot_models
        if already_exists:
            custom_model_id = self.datarobot_models[model_info.git_model_id].model["id"]
        else:
            custom_model_id = self._dr_client.create_custom_model(model_info)
            logger.info(f"Custom inference model was created: {custom_model_id}")
        return custom_model_id, already_exists

    def _create_custom_model_version(self, custom_model_id, model_info):
        if model_info.should_upload_all_files:
            changed_file_paths = list(model_info.model_file_paths.values())
        else:
            changed_file_paths = model_info.changed_or_new_files

        logger.info(
            "Create custom inference model version. git_model_id: "
            f" {model_info.git_model_id}, from_latest: {model_info.should_upload_all_files}"
        )
        logger.debug(
            f"Files to be uploaded: {changed_file_paths}, git_model_id: {model_info.git_model_id}"
        )

        if self.is_pull_request:
            if self._repo.num_remotes() == 0:
                # Only to support the functional tests, which do not have a remote repository.
                main_branch = self.options.branch
            else:
                # This is the expected path when working against a remote GitHub repository.
                main_branch = f"{self._repo.remote_name()}/{self.options.branch}"
            main_branch_commit_sha = self._repo.merge_base_commit_sha(main_branch, self.github_sha)
            pull_request_commit_sha = self._repo.feature_branch_top_commit_sha_of_a_merge_commit(
                self.github_sha
            )
            commit_url = GitTool.GITHUB_COMMIT_URL_PATTERN.format(
                user_and_project=self.github_repository, sha=pull_request_commit_sha
            )
        else:
            main_branch_commit_sha = self.github_sha
            pull_request_commit_sha = None
            commit_url = GitTool.GITHUB_COMMIT_URL_PATTERN.format(
                user_and_project=self.github_repository, sha=main_branch_commit_sha
            )
        logger.info(
            f"GitHub commit URL: {commit_url}, main branch commit sha: {main_branch_commit_sha}, "
            f"pull request commit sha: {pull_request_commit_sha}"
        )

        return self._dr_client.create_custom_model_version(
            custom_model_id,
            model_info,
            commit_url,
            main_branch_commit_sha,
            pull_request_commit_sha,
            changed_file_paths,
            model_info.deleted_file_ids,
            from_latest=not model_info.should_upload_all_files,
        )

    def _test_custom_model_version(self, model_id, model_version_id, model_info):
        logger.info("Executing custom model test ...")
        self._dr_client.run_custom_model_version_testing(model_id, model_version_id, model_info)

    def _handle_deleted_models(self):
        if not self.options.allow_model_deletion:
            logger.info("Skip handling models deletion because it is not enabled.")
            return

        missing_locally_id_to_git_id = {}
        for git_model_id, datarobot_model in self.datarobot_models.items():
            if git_model_id not in self.models_info:
                missing_locally_id_to_git_id[datarobot_model.model["id"]] = git_model_id

        if missing_locally_id_to_git_id:
            model_ids_to_fetch = list(missing_locally_id_to_git_id.keys())
            deployments = self._dr_client.fetch_custom_model_deployments(model_ids_to_fetch)
            if self.is_pull_request:
                # Only check that deleted models are not deployed in DataRobot. If so, make sure to
                # report a failure.
                self._validate_that_model_to_be_deleted_is_not_deployed(
                    missing_locally_id_to_git_id, deployments
                )
            else:
                self._actually_delete_models(missing_locally_id_to_git_id, deployments)

    @staticmethod
    def _validate_that_model_to_be_deleted_is_not_deployed(
        missing_locally_id_to_git_id, deployments
    ):
        # Only check that deleted models are not deployed in DataRobot. If so, make sure to
        # report a failure.
        logger.info("Detecting that deployed models are not deleted ...")
        if deployments:
            # TODO: do not raise an error for 'dirty' models, because anyway these should not be
            #  deleted.
            msg = ""
            for deployment in deployments:
                model_id = deployment["customModel"]["id"]
                msg += (
                    f"Deployment: {deployment['id']}, "
                    f"model_id: {model_id}, "
                    f"git_model_id: {missing_locally_id_to_git_id[model_id]}"
                    "\n"
                )
            raise IllegalModelDeletion(
                f"Models cannot be deleted because of existing deployments.\n{msg}"
            )

    def _actually_delete_models(self, missing_locally_id_to_git_id, deployments):
        logger.info("Deleting models ...")
        for model_id, git_model_id in missing_locally_id_to_git_id.items():
            # TODO: skip deletion of 'dirty' models. Only show a warning.
            if any(model_id == deployment["customModel"]["id"] for deployment in deployments):
                logger.warning(
                    f"Skipping model deletion because it is deployed. "
                    f"git_model_id: {git_model_id}, model_id: {model_id}"
                )
                continue
            try:
                self._dr_client.delete_custom_model_by_model_id(model_id)
                self._total_deleted += 1
                self._total_affected += 1
                logger.info(
                    f"Model was deleted with success. git_model_id: {git_model_id}, "
                    f"model_id: {model_id}"
                )
            except DataRobotClientError as ex:
                logger.error(str(ex))
