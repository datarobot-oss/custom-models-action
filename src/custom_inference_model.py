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
from common.exceptions import (
    ModelMainEntryPointNotFound,
    DataRobotClientError,
    IllegalModelDeletion,
    ModelMetadataAlreadyExists,
)
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
        os.environ["GIT_PYTHON_TRACE"] = "full"
        self._repo = GitTool(self.options.root_dir)
        self._models_info = []
        self._datarobot_models = {}
        self._datarobot_models_by_id = {}
        self._dr_client = DrClient(
            self.options.webserver,
            self.options.api_token,
            verify_cert=not self.options.skip_cert_verification,
        )
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

        if not self._prerequisites():
            return

        self._scan_and_load_models_metadata()
        self._run()

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

        num_commits = self._repo.num_commits()
        if num_commits < 2:
            logger.warning(
                "Skip custom inference model action. The minimum number of commits should be 2. "
                f"Current number is {num_commits}"
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
        try:
            already_exists = next(
                m for m in self._models_info if model_info.git_model_id == m.git_model_id
            )
            raise ModelMetadataAlreadyExists(
                f"Model {model_info.git_model_id} already exists. "
                f"New model yaml path: {model_info.yaml_filepath}. "
                f"Existing model yaml path: {already_exists.yaml_filepath}."
            )
        except StopIteration:
            pass

        logger.info(
            f"Adding new model metadata. Git model ID: {model_info.git_model_id}. "
            f"Model metadata yaml path: {model_info.yaml_filepath}."
        )
        self._models_info.append(model_info)

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

    def _get_latest_provisioned_model_git_version(self, model_info):
        latest_version = self.datarobot_models[model_info.git_model_id].latest_version
        return latest_version["gitModelVersion"]

    @abstractmethod
    def _run(self):
        pass


class CustomInferenceModel(CustomInferenceModelBase):
    class RelativeTo(Enum):
        ROOT = 1
        MODEL = 2

    def __init__(self, options):
        super().__init__(options)
        self._total_affected_models = 0
        self._total_created_models = 0
        self._total_deleted_models = 0

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
                ::set-output name=total-affected-models::{self._total_affected_models}
                ::set-output name=total-created-models::{self._total_created_models}
                ::set-output name=total-deleted-models::{self._total_deleted_models}
                """
            )

    def _collect_datarobot_model_files(self):
        logger.info("Collecting DataRobot model files ...")
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

    def _lookup_affected_models_by_the_current_action(self):
        logger.info("Lookup affected models by the current commit ...")

        for model_info in self.models_info:
            model_info.changed_or_new_files = []
            model_info.deleted_file_ids = []
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
        ancestor_sha = self._get_latest_provisioned_model_git_version(model_info)[
            self.ancestor_attribute_ref
        ]
        if not ancestor_sha:
            # Either the model has never provisioned of the user created a version with a non
            # GitHub action client.
            return False

        # Users may have few local commits between remote pushes
        return self._repo.is_ancestor_of(ancestor_sha, self.github_sha)

    def _lookup_affected_models(self):
        # In a PR a merge commit is always the last commit, which we need to ignore.
        if logger.isEnabledFor(logging.DEBUG):
            self._repo.print_pretty_log()

        for model_info in self.models_info:
            if model_info.should_upload_all_files:
                continue
            from_commit_sha = self._get_latest_provisioned_model_git_version(model_info)[
                self.ancestor_attribute_ref
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
            if model_info.git_model_id in self.datarobot_models:
                latest_version = self.datarobot_models[model_info.git_model_id].latest_version
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

    def _apply_datarobot_actions_for_affected_models(self):
        logger.info("Apply DataRobot actions for affected models ...")
        self._handle_model_changes_or_creation()
        self._handle_deleted_models()

    def _handle_model_changes_or_creation(self):
        for model_info in self._models_info:
            if model_info.is_affected_by_commit:
                self._total_affected_models += 1
                logger.info(f"Model '{model_info.model_path}' is affected by commit.")

                custom_model_id, already_existed = self._get_or_create_custom_model(model_info)
                if not already_existed:
                    self._total_created_models += 1
                version_id = self._create_custom_model_version(custom_model_id, model_info)
                if model_info.should_run_test:
                    self._test_custom_model_version(custom_model_id, version_id, model_info)

                logger.info(
                    "Custom inference model version was successfully created. "
                    f"git_model_id: {model_info.git_model_id}, model_id: {custom_model_id}, "
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
            changed_files_info = self._get_relative_paths(model_info, model_info.model_file_paths)
        else:
            changed_files_info = self._get_relative_paths(
                model_info, model_info.changed_or_new_files
            )

        logger.info(
            "Create custom inference model version. git_model_id: "
            f" {model_info.git_model_id}, from_latest: {model_info.should_upload_all_files}"
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

    def _handle_deleted_models(self):
        if not self.options.allow_model_deletion:
            logger.info("Skip handling models deletion because it is not enabled.")
            return

        missing_locally_id_to_git_id = {}
        for git_model_id, datarobot_model in self.datarobot_models.items():
            if all(git_model_id != model_info.git_model_id for model_info in self._models_info):
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
                msg += (
                    f"Deployment: {deployment['id']}, "
                    f"model_id: {deployment['customModel']['id']}, "
                    f"git_model_id: {missing_locally_id_to_git_id[deployment['customModel']['id']]}"
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
                self._total_deleted_models += 1
                logger.info(
                    f"Model was deleted with success. git_model_id: {git_model_id}, "
                    f"model_id: {model_id}"
                )
            except DataRobotClientError as ex:
                logger.error(str(ex))
