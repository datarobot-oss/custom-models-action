#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
This module controls and coordinate between local model definitions and DataRobot models.
In highlights, it scans and loads model definitions from the local source tree, perform
validations and then applies actions in DataRobot.
"""

import logging
import os
import re
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict

import yaml

from common.data_types import DataRobotModel
from common.exceptions import DataRobotClientError
from common.exceptions import IllegalModelDeletion
from common.exceptions import ModelMainEntryPointNotFound
from common.exceptions import ModelMetadataAlreadyExists
from common.exceptions import SharedAndLocalPathCollision
from common.exceptions import UnexpectedResult
from common.git_tool import GitTool
from common.github_env import GitHubEnv
from dr_client import DrClient
from model_file_path import ModelFilePath
from model_info import ModelInfo
from schema_validator import ModelSchema

logger = logging.getLogger()


class ControllerBase(ABC):
    """
    A base class that contains attributes and methods that will be shared by the custom
    inference model and deployment classes.
    """

    MODELS_LABEL = "models"
    DEPLOYMENTS_LABEL = "deployments"

    _models_info: Dict[str, ModelInfo]

    @dataclass
    class Stats:
        """Contains statistics attributes that will be exposed by the GitHub actions."""

        total_affected: int = 0
        total_created: int = 0
        total_deleted: int = 0
        total_created_versions: int = 0

        def save(self, label):
            """Save the statistics to the GitHub environment."""

            GitHubEnv.set_output_param(f"total-affected-{label}", self.total_affected)
            GitHubEnv.set_output_param(f"total-created-{label}", self.total_created)
            GitHubEnv.set_output_param(f"total-deleted-{label}", self.total_deleted)
            if label == ControllerBase.MODELS_LABEL:
                GitHubEnv.set_output_param(
                    "total-created-model-versions", self.total_created_versions
                )

    def __init__(self, options, repo):
        self._options = options
        self._workspace_path = GitHubEnv.workspace_path()
        self._repo = repo
        self._dr_client = DrClient(
            self.options.webserver,
            self.options.api_token,
            verify_cert=not self.options.skip_cert_verification,
        )
        self._stats = self.Stats()
        logger.info(
            "GITHUB_EVENT_NAME: %s, GITHUB_SHA: %s, GITHUB_REPOSITORY: %s, GITHUB_REF_NAME: %s",
            GitHubEnv.event_name(),
            GitHubEnv.github_sha(),
            GitHubEnv.github_repository(),
            GitHubEnv.ref_name(),
        )

    @property
    def options(self):
        """The command line argument values."""

        return self._options

    @property
    def stats(self):
        """A property to return the Stats instance."""

        return self._stats

    def _next_yaml_content_in_repo(self):
        yaml_files = glob(f"{self._workspace_path}/**/*.yaml", recursive=True)
        yaml_files.extend(glob(f"{self._workspace_path}/**/*.yml", recursive=True))
        for yaml_path in yaml_files:
            with open(yaml_path, encoding="utf-8") as fd:
                yaml_content = yaml.safe_load(fd)
                if not yaml_content:
                    logger.warning("Detected an invalid or empty yaml file: %s", yaml_path)
                else:
                    yield yaml_path, yaml_content

    @staticmethod
    def _make_directory_pattern_recursive(pattern):
        return f"{pattern}**" if pattern.endswith("/") else pattern

    def _to_absolute(self, path, parent):
        match = re.match(r"^(/|\$ROOT/)", path)
        if match:
            path = path.replace(match[0], "", 1)
            path = f"{self._workspace_path}/{path}"
        else:
            path = f"{parent}/{path}"
        return path

    def save_statistics(self):
        """Save the statistics that are configured by the GitHub action."""

        self.stats.save(self._label())

    @abstractmethod
    def _label(self):
        pass


class ModelController(ControllerBase):
    """A custom inference model implementation of the GitHub action"""

    def __init__(self, options, repo):
        super().__init__(options, repo)
        self._models_info = {}
        self._datarobot_models = {}
        self._datarobot_models_by_id = {}

    def _label(self):
        return self.MODELS_LABEL

    @property
    def models_info(self):
        """A list of model info entities that were loaded from the local repository"""
        return self._models_info

    @property
    def datarobot_models(self):
        """A list of DataRobot model entities that were fetched from DataRobot"""
        return self._datarobot_models

    def datarobot_model_by_id(self, model_id):
        """
        A DataRobot model entity of a given model ID.

        Parameters
        ----------
        model_id : str
            The model ID.

        Returns
        -------
        dict,
            A DataRobot model entity.
        """

        return self._datarobot_models_by_id.get(model_id)

    def scan_and_load_models_metadata(self):
        """Scan and load model metadata yaml files from the local source tree."""

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

    def _add_new_model_info(self, model_info):
        if model_info.user_provided_id in self.models_info:
            raise ModelMetadataAlreadyExists(
                f"Model {model_info.user_provided_id} already exists. "
                f"New model yaml path: {model_info.yaml_filepath}."
            )

        logger.info(
            "Adding new model metadata. User provided ID: %s. Model metadata yaml path: %s.",
            model_info.user_provided_id,
            model_info.yaml_filepath,
        )
        self.models_info[model_info.user_provided_id] = model_info

    def collect_datarobot_model_files(self):
        """Collect model definition yaml files from local source tree."""

        logger.info("Collecting DataRobot model files ...")
        for _, model_info in self.models_info.items():
            include_glob_patterns = model_info.metadata[ModelSchema.VERSION_KEY][
                ModelSchema.INCLUDE_GLOB_KEY
            ]
            included_paths = set([])
            if include_glob_patterns:
                for include_glob_pattern in include_glob_patterns:
                    include_glob_pattern = self._make_directory_pattern_recursive(
                        include_glob_pattern
                    )
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
                exclude_glob_pattern = self._make_directory_pattern_recursive(exclude_glob_pattern)
                exclude_glob_pattern = self._to_absolute(
                    exclude_glob_pattern, model_info.model_path
                )
                # For excluded directories always assume recursive
                if Path(exclude_glob_pattern).is_dir():
                    exclude_glob_pattern += "/**"

                excluded_paths.update(glob(exclude_glob_pattern, recursive=True))

            self._set_filtered_model_paths(
                model_info, included_paths, excluded_paths, self._workspace_path
            )
            self._validate_model_integrity(model_info)
            logger.info("Model %s detected and verified.", model_info.model_path)

    @classmethod
    def _set_filtered_model_paths(cls, model_info, included_paths, excluded_paths, workspace_path):
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

        model_info.set_model_paths(final_model_paths, workspace_path)

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
                f"Model (Id: {model_info.user_provided_id}) main entry point not found ("
                f"custom.py).\nExisting files: {model_info.model_file_paths}"
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
                "Invalid file tree. Shared file(s)/package(s) collide with local model's "
                f"file(s)/package(s). Collisions: {collisions}."
            )

    def fetch_models_from_datarobot(self):
        """Retrieve custom inference models from DataRobot."""

        logger.info("Fetching models from DataRobot ...")
        custom_inference_models = self._dr_client.fetch_custom_models()
        for custom_model in custom_inference_models:
            user_provided_id = custom_model.get("userProvidedId")
            if user_provided_id:
                datarobot_model_id = custom_model["id"]
                model_versions = self._dr_client.fetch_custom_model_versions(
                    datarobot_model_id, json={"limit": 1}
                )
                latest_version = model_versions[0] if model_versions else None
                if not latest_version:
                    logger.warning(
                        "Model exists without a version! user_provided_id: %s, custom_model_id: %s",
                        user_provided_id,
                        datarobot_model_id,
                    )
                self._set_datarobot_custom_model(user_provided_id, custom_model, latest_version)

    def _set_datarobot_custom_model(self, user_provided_id, custom_model, latest_version=None):
        datarobot_model = DataRobotModel(model=custom_model, latest_version=latest_version)
        self.datarobot_models[user_provided_id] = datarobot_model
        self._datarobot_models_by_id[custom_model["id"]] = datarobot_model

    def lookup_affected_models_by_the_current_action(self):
        """Search for models that were affected byt the current commit."""

        logger.info("Lookup affected models by the current commit ...")

        for _, model_info in self.models_info.items():
            should_upload_all_files = self._should_upload_all_files(model_info)
            model_info.flags = model_info.Flags(
                should_upload_all_files=should_upload_all_files,
                should_update_settings=should_upload_all_files,
            )

        self._lookup_affected_models()

    def _should_upload_all_files(self, model_info):
        return not self._model_version_exists(model_info) or not self._valid_ancestor(model_info)

    def _model_version_exists(self, model_info):
        return (
            model_info.user_provided_id in self.datarobot_models
            and self.datarobot_models[model_info.user_provided_id].latest_version
        )

    def _valid_ancestor(self, model_info):
        ancestor_sha = self._get_latest_model_version_git_commit_ancestor(model_info)
        if not ancestor_sha:
            return False

        # Users may have few local commits between remote pushes
        is_ancestor = self._repo.is_ancestor_of(ancestor_sha, GitHubEnv.github_sha())
        logger.debug(
            "Is the latest model version's git commit sha (%s) "
            "ancestor of the current commit (%s)? Answer: %s",
            ancestor_sha,
            GitHubEnv.github_sha(),
            is_ancestor,
        )
        return is_ancestor

    def _get_latest_model_version_git_commit_ancestor(self, model_info):
        latest_version = self.datarobot_models[model_info.user_provided_id].latest_version
        git_model_version = latest_version.get("gitModelVersion")
        if not git_model_version:
            # Either the model has never provisioned of the user created a version with a non
            # GitHub action client.
            return False

        return git_model_version[self.ancestor_attribute_ref(git_model_version)]

    @staticmethod
    def ancestor_attribute_ref(git_model_version):
        """
        The attribute name from DataRobot public API that should be taken into account, depending
        on the event that triggered the workflow and a reference name that is associated
        with the latest custom model version.

        Parameters
        ----------
        git_model_version : dict
            The GitModelVersion from a DataRobot custom inference model version entity.

        Returns
        -------
        str,
            The DataRobot public API attribute name.
        """

        latest_custom_model_version_ref_name = git_model_version["refName"]
        if (
            GitHubEnv.is_pull_request()
            and GitHubEnv.ref_name() == latest_custom_model_version_ref_name
        ):
            return "pullRequestCommitSha"
        return "mainBranchCommitSha"

    def _lookup_affected_models(self):
        # In a PR a merge commit is always the last commit, which we need to ignore.
        if logger.isEnabledFor(logging.DEBUG):
            self._repo.print_pretty_log()

        for _, model_info in self.models_info.items():
            model_info.file_changes = ModelInfo.FileChanges()

            logger.debug(
                "Searching model %s changes since its last DR version.", model_info.user_provided_id
            )
            if model_info.flags.should_upload_all_files:
                logger.debug("Model %s will upload all its files.", model_info.user_provided_id)
                continue

            from_commit_sha = self._get_latest_model_version_git_commit_ancestor(model_info)
            if not from_commit_sha:
                raise UnexpectedResult(
                    "Unexpected None ancestor commit sha, "
                    f"model_git_id: {model_info.user_provided_id}"
                )
            changed_files, deleted_files = self._repo.find_changed_files(
                GitHubEnv.github_sha(), from_commit_sha
            )
            logger.debug(
                "Model %s was changed since commit %s. Changed files: %s. Deleted files: %s.",
                model_info.user_provided_id,
                from_commit_sha,
                changed_files,
                deleted_files,
            )
            self._handle_changed_or_new_files(model_info, changed_files)
            self._handle_deleted_files(model_info)

    @classmethod
    def _handle_changed_or_new_files(cls, model_info, changed_or_new_files):
        for changed_file in changed_or_new_files:
            changed_model_filepath = model_info.model_file_paths.get(changed_file)
            if changed_model_filepath:
                logger.debug(
                    "Changed/new file '%s' affects model '%s'",
                    changed_file,
                    model_info.model_path.name,
                )
                model_info.file_changes.add_changed(changed_model_filepath)

            cls._mark_changes_in_model_settings(model_info, changed_file)

    @classmethod
    def _mark_changes_in_model_settings(cls, model_info, changed_file):
        # A change may happen in the model's definition (yaml), which is not necessarily
        # included by the glob patterns
        if any(changed_file.suffix == suffix for suffix in [".yaml", ".yml"]):
            logger.debug(
                "Check if changes were made to the model's yaml file. model's yaml "
                "file: %s, checked yaml file: %s",
                model_info.yaml_filepath,
                changed_file,
            )
            if model_info.yaml_filepath.samefile(changed_file):
                logger.debug("The model's settings were changed. path %s", model_info.model_path)
                model_info.flags.should_update_settings = True

    def _handle_deleted_files(self, model_info):
        """
        Deleted files may happen in two scenarios:
        - A file that belongs to the model was deleted from the source tree
        - A file that belonged to the model was excluded using the 'glob' pattern

        The logic to tackle these both scenarios is to go over the current files in DataRobot
        model version and check if the file exists in the model info. If not, it is regarded
        as deleted.
        """

        if model_info.user_provided_id in self.datarobot_models:
            latest_version = self.datarobot_models[model_info.user_provided_id].latest_version
            if latest_version:
                for item in latest_version["items"]:
                    if not self._file_path_belongs_to_model(item["filePath"], model_info):
                        model_info.file_changes.add_deleted_file_id(item["id"])
                        logger.debug(
                            "File ID was added to deleted list. Model: %s, path: %s",
                            model_info.model_path,
                            item["filePath"],
                        )

    def _file_path_belongs_to_model(self, path_under_model_to_check, model_info):
        for local_file_path in model_info.model_file_paths:
            existing_path_under_model, _ = ModelFilePath.get_path_under_model(
                local_file_path, model_info.model_path, self._workspace_path
            )
            if path_under_model_to_check == existing_path_under_model:
                return True
        return False

    def handle_model_changes(self):
        """Apply changes in DataRobot for models that were affected by the current commit."""

        for user_provided_id, model_info in self.models_info.items():
            already_exists = user_provided_id in self.datarobot_models
            custom_model = self.datarobot_models[user_provided_id].model if already_exists else None
            latest_version = (
                self.datarobot_models[user_provided_id].latest_version if already_exists else None
            )

            if model_info.is_affected_by_commit(latest_version):
                logger.info("Model '%s' is affected by commit.", model_info.model_path)

                if model_info.should_create_new_version(latest_version):
                    if not custom_model:
                        custom_model = self._create_custom_model(model_info)
                        self.stats.total_created += 1

                    custom_model_id = custom_model["id"]
                    latest_version = self._create_custom_model_version(custom_model_id, model_info)
                    self.datarobot_models[user_provided_id].latest_version = latest_version

                    self.stats.total_created_versions += 1
                    logger.info(
                        "Custom inference model version was successfully created. "
                        "user_provided_id: %s, model_id: %s, version_id: %s.",
                        user_provided_id,
                        custom_model_id,
                        latest_version["id"],
                    )
                else:
                    if GitHubEnv.is_push():
                        # Upon pushing to the main branch, If the model was somehow affected by the
                        # given event, make sure to update the main branch commit SHA. This solves
                        # an issue of follow-up pull requests, which are not related to the given
                        # model that should not affect the given model.
                        latest_version = self._update_custom_model_version_git_attributes(
                            model_info, latest_version
                        )
                        self.datarobot_models[user_provided_id].latest_version = latest_version

                if model_info.flags.should_update_settings:
                    self._update_settings(custom_model, model_info)

                if model_info.should_run_test:
                    self._test_custom_model_version(
                        custom_model["id"], latest_version["id"], model_info
                    )

                self.stats.total_affected += 1

    def _update_custom_model_version_git_attributes(self, model_info, custom_model_version):
        main_branch_commit_sha = GitHubEnv.github_sha()
        commit_url = GitTool.GITHUB_COMMIT_URL_PATTERN.format(
            user_and_project=GitHubEnv.github_repository(),
            sha=main_branch_commit_sha,
        )
        ref_name = GitHubEnv.ref_name()
        logger.info(
            "Updating model git info. path: %s, commit_sha: %s, commit_url: %s, ref_name: %s",
            model_info.model_path,
            main_branch_commit_sha,
            commit_url,
            ref_name,
        )
        return self._dr_client.update_custom_model_version_main_branch_commit_sha(
            custom_model_version, main_branch_commit_sha, commit_url, ref_name
        )

    def _create_custom_model(self, model_info):
        custom_model = self._dr_client.create_custom_model(model_info)
        self._set_datarobot_custom_model(model_info.user_provided_id, custom_model)
        logger.info("Custom inference model was created: %s", custom_model["id"])
        return custom_model

    def _create_custom_model_version(self, custom_model_id, model_info):
        if model_info.flags.should_upload_all_files:
            changed_file_paths = list(model_info.model_file_paths.values())
        else:
            changed_file_paths = model_info.file_changes.changed_or_new_files

        logger.info(
            "Create custom inference model version. user_provided_id:  %s, from_latest: %s",
            model_info.user_provided_id,
            model_info.flags.should_create_version_from_latest,
        )
        logger.debug(
            "Files to be uploaded: %s, user_provided_id: %s",
            [p.under_model for p in changed_file_paths],
            model_info.user_provided_id,
        )

        if GitHubEnv.is_pull_request():
            if self._repo.num_remotes() == 0:
                # Only to support the functional tests, which do not have a remote repository.
                main_branch = self.options.branch
            else:
                # This is the expected path when working against a remote GitHub repository.
                main_branch = f"{self._repo.remote_name()}/{self.options.branch}"
            main_branch_commit_sha = self._repo.merge_base_commit_sha(
                main_branch, GitHubEnv.github_sha()
            )
            pull_request_commit_sha = self._repo.feature_branch_top_commit_sha_of_a_merge_commit(
                GitHubEnv.github_sha()
            )
            commit_url = GitTool.GITHUB_COMMIT_URL_PATTERN.format(
                user_and_project=GitHubEnv.github_repository(), sha=pull_request_commit_sha
            )
        else:
            main_branch_commit_sha = GitHubEnv.github_sha()
            pull_request_commit_sha = None
            commit_url = GitTool.GITHUB_COMMIT_URL_PATTERN.format(
                user_and_project=GitHubEnv.github_repository(), sha=main_branch_commit_sha
            )
        logger.info(
            "GitHub version info. Ref name: %s, commit URL: %s, main branch commit sha: %s, "
            "pull request commit sha: %s",
            GitHubEnv.ref_name(),
            commit_url,
            main_branch_commit_sha,
            pull_request_commit_sha,
        )

        custom_model_version = self._dr_client.create_custom_model_version(
            custom_model_id,
            model_info,
            GitHubEnv.ref_name(),
            commit_url,
            main_branch_commit_sha,
            pull_request_commit_sha,
            changed_file_paths,
            model_info.file_changes.deleted_file_ids,
            from_latest=model_info.flags.should_create_version_from_latest,
        )
        return custom_model_version

    def _test_custom_model_version(self, model_id, model_version_id, model_info):
        logger.info("Executing custom model test ...")
        self._dr_client.run_custom_model_version_testing(model_id, model_version_id, model_info)

    def _update_settings(self, datarobot_custom_model, model_info):
        self._update_model_settings(datarobot_custom_model, model_info)
        # NOTE: training/holdout datasets update should always come after model's setting update
        self._update_training_and_holdout_datasets(datarobot_custom_model, model_info)

    def _update_training_and_holdout_datasets(self, datarobot_custom_model, model_info):
        if model_info.is_unstructured:
            custom_model = (
                self._dr_client.update_training_and_holdout_datasets_for_unstructured_models(
                    datarobot_custom_model, model_info
                )
            )
            if custom_model:
                logger.info(
                    "Training / holdout dataset were updated for unstructured model. "
                    "User provided ID: %s. Training dataset name: %s. Training dataset ID: %s. "
                    "Holdout dataset name: %s. Holdout dataset ID: %s",
                    model_info.user_provided_id,
                    custom_model["externalMlopsStatsConfig"]["trainingDatasetName"],
                    custom_model["externalMlopsStatsConfig"]["trainingDatasetId"],
                    custom_model["externalMlopsStatsConfig"]["holdoutDatasetName"],
                    custom_model["externalMlopsStatsConfig"]["holdoutDatasetId"],
                )
        else:
            custom_model = self._dr_client.update_training_dataset_for_structured_models(
                datarobot_custom_model, model_info
            )
            if custom_model:
                logger.info(
                    "Training dataset was updated for structured model. User provided ID: %s. "
                    "Dataset name: %s. Dataset ID: %s. Dataset version ID: %s.",
                    model_info.user_provided_id,
                    custom_model["trainingDataFileName"],
                    custom_model["trainingDatasetId"],
                    custom_model["trainingDatasetVersionId"],
                )

    def _update_model_settings(self, datarobot_custom_model, model_info):
        custom_model = self._dr_client.update_model_settings(datarobot_custom_model, model_info)
        if custom_model:
            logger.info(
                "Model settings were updated. User provided ID: %s.", model_info.user_provided_id
            )

    def handle_deleted_models(self):
        """
        Delete models in DataRobot for all deleted model definitions in the current commit.
        The actual deletion takes place only in a push event. For a pull request event there's
        only a validation.
        """

        missing_locally_id_to_git_id = {}
        for user_provided_id, datarobot_model in self.datarobot_models.items():
            if user_provided_id not in self.models_info:
                missing_locally_id_to_git_id[datarobot_model.model["id"]] = user_provided_id

        if missing_locally_id_to_git_id:
            if not self.options.allow_model_deletion:
                missing_user_provided_ids = list(missing_locally_id_to_git_id.values())
                raise IllegalModelDeletion(
                    "Model deletion was configured as not being allowed. "
                    f"The missing models in the local source tree are: {missing_user_provided_ids}"
                )

            model_ids_to_fetch = list(missing_locally_id_to_git_id.keys())
            deployments = self._dr_client.fetch_custom_model_deployments(model_ids_to_fetch)

            self._validate_that_model_to_be_deleted_is_not_deployed(
                missing_locally_id_to_git_id, deployments
            )

            if GitHubEnv.is_push():
                self._actually_delete_models(missing_locally_id_to_git_id, deployments)

    @staticmethod
    def _validate_that_model_to_be_deleted_is_not_deployed(
        missing_locally_id_to_git_id, deployments
    ):
        # Only check that deleted models are not deployed in DataRobot. If so, make sure to
        # report a failure.
        logger.info("Detecting that deployed models are not deleted ...")
        if deployments:
            msg = ""
            for deployment in deployments:
                model_id = deployment["customModel"]["id"]
                msg += (
                    f"Deployment: {deployment['id']}, "
                    f"model_id: {model_id}, "
                    f"user_provided_id: {missing_locally_id_to_git_id.get(model_id)}"
                    "\n"
                )
            raise IllegalModelDeletion(
                f"Models cannot be deleted because of existing deployments.\n{msg}"
            )

    def _actually_delete_models(self, missing_locally_id_to_git_id, deployments):
        logger.info("Deleting models ...")
        for model_id, user_provided_id in missing_locally_id_to_git_id.items():
            if any(model_id == deployment["customModel"]["id"] for deployment in deployments):
                logger.warning(
                    "Skipping model deletion because it is deployed. user_provided_id: %s, "
                    "model_id: %s",
                    user_provided_id,
                    model_id,
                )
                continue
            try:
                self._dr_client.delete_custom_model_by_model_id(model_id)
                self.stats.total_deleted += 1
                self.stats.total_affected += 1
                logger.info(
                    "Model was deleted with success. user_provided_id: %s, model_id: %s",
                    user_provided_id,
                    model_id,
                )
            except DataRobotClientError as ex:
                logger.error(str(ex))
