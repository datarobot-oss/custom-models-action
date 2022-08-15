#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
The implementation of the custom inference model GitHub action. In highlights, it scans
and loads model definitions from the local source tree, performs validations and then detects
which models were affected by the last Git action and applies the proper actions in DataRobot
application.
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
from dr_client import DrClient
from model_file_path import ModelFilePath
from model_info import ModelInfo
from schema_validator import ModelSchema

logger = logging.getLogger()


class CustomInferenceModelBase(ABC):
    """
    A base class that contains attributes and methods that will be shared by the custom
    inference model and deployment classes.
    """

    _models_info: Dict[str, ModelInfo]

    @dataclass
    class Stats:
        """Contains statistics attributes that will be exposed by the GitHub actions."""

        total_affected: int = 0
        total_created: int = 0
        total_deleted: int = 0

        def print(self, label):
            """
            Print the statistics to the standard output. This is how the GitHub action
            exposes statistics to other steps in the workflow.
            """

            print(
                f"""
                ::set-output name=total-affected-{label}::{self.total_affected}
                ::set-output name=total-created-{label}::{self.total_created}
                ::set-output name=total-deleted-{label}::{self.total_deleted}
                """
            )

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
        self._stats = self.Stats()
        logger.info(
            "GITHUB_EVENT_NAME: %s, GITHUB_SHA: %s, GITHUB_REPOSITORY: %s, GITHUB_REF_NAME: %s",
            self.event_name,
            self.github_sha,
            self.github_repository,
            self.ref_name,
        )

    @property
    def options(self):
        """The command line argument values"""
        return self._options

    @property
    def event_name(self):
        """The event name that triggered the GitHub workflow."""
        return os.environ.get("GITHUB_EVENT_NAME")

    @property
    def github_sha(self):
        """The commit SHA that triggered the GitHub workflow"""
        return os.environ.get("GITHUB_SHA")

    @property
    def github_repository(self):
        """The owner and repository name from GtiHub"""
        return os.environ.get("GITHUB_REPOSITORY")

    @property
    def ref_name(self):
        """The branch or tag name that triggered the GitHub workflow run."""
        return os.environ.get("GITHUB_REF_NAME")

    @property
    def is_pull_request(self):
        """Whether the event that triggered the GitHub workflow is a pull request."""
        return self.event_name == "pull_request"

    @property
    def is_push(self):
        """Whether the event that triggered the GitHub workflow is a push."""
        return self.event_name == "push"

    def ancestor_attribute_ref(self, git_model_version):
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
        if self.is_pull_request and self.ref_name == latest_custom_model_version_ref_name:
            return "pullRequestCommitSha"
        return "mainBranchCommitSha"

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

    def run(self):
        """Executes the GitHub action logic to manage custom inference models"""

        try:
            if not self._prerequisites():
                return

            self._scan_and_load_models_metadata()
            self._run()
        finally:
            self._stats.print(self._label())

    def _prerequisites(self):
        supported_events = ["push", "pull_request"]
        if self.event_name not in supported_events:
            logger.warning(
                "Skip custom inference model action. It is expected to be executed only "
                "on %s events. Current event: %s.",
                supported_events,
                self.event_name,
            )
            return False

        base_ref = os.environ.get("GITHUB_BASE_REF")
        logger.info("GITHUB_BASE_REF: %s.", base_ref)
        if self.is_pull_request and base_ref != self.options.branch:
            logger.info(
                "Skip custom inference model action. It is executed only when the referenced "
                "branch is %s. Current ref branch: %s.",
                self.options.branch,
                base_ref,
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
                    "should be 2. Current number is %d.",
                    num_commits,
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
            with open(yaml_path, encoding="utf-8") as fd:
                yield yaml_path, yaml.safe_load(fd)

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
                f"New model yaml path: {model_info.yaml_filepath}."
            )

        logger.info(
            "Adding new model metadata. Git model ID: %s. Model metadata yaml path: %s.",
            model_info.git_model_id,
            model_info.yaml_filepath,
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
                        "Model exists without a version! git_model_id: %s, custom_model_id: %s",
                        git_model_id,
                        datarobot_model_id,
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

        return git_model_version[self.ancestor_attribute_ref(git_model_version)]

    @abstractmethod
    def _label(self):
        pass

    @abstractmethod
    def _run(self):
        pass


class CustomInferenceModel(CustomInferenceModelBase):
    """A custom inference model implementation of the GitHub action"""

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

        self._collect_datarobot_model_files()
        self._fetch_models_from_datarobot()
        self._lookup_affected_models_by_the_current_action()
        self._apply_datarobot_actions_for_affected_models()

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
            logger.info("Model %s detected and verified.", model_info.model_path)

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
                f"Model (Id: {model_info.git_model_id}) main entry point not found ("
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

    def _lookup_affected_models_by_the_current_action(self):
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
            model_info.git_model_id in self.datarobot_models
            and self.datarobot_models[model_info.git_model_id].latest_version
        )

    def _valid_ancestor(self, model_info):
        ancestor_sha = self._get_latest_model_version_git_commit_ancestor(model_info)
        if not ancestor_sha:
            return False

        # Users may have few local commits between remote pushes
        is_ancestor = self._repo.is_ancestor_of(ancestor_sha, self.github_sha)
        logger.debug(
            "Is the latest model version's git commit sha (%s) "
            "ancestor of the current commit (%s)? Answer: %s",
            ancestor_sha,
            self.github_sha,
            is_ancestor,
        )
        return is_ancestor

    def _lookup_affected_models(self):
        # In a PR a merge commit is always the last commit, which we need to ignore.
        if logger.isEnabledFor(logging.DEBUG):
            self._repo.print_pretty_log()

        for _, model_info in self.models_info.items():
            model_info.file_changes = ModelInfo.FileChanges()

            logger.debug(
                "Searching model %s changes since its last DR version.", model_info.git_model_id
            )
            if model_info.flags.should_upload_all_files:
                logger.debug("Model %s will upload all its files.", model_info.git_model_id)
                continue

            from_commit_sha = self._get_latest_model_version_git_commit_ancestor(model_info)
            if not from_commit_sha:
                raise UnexpectedResult(
                    f"Unexpected None ancestor commit sha, model_git_id: {model_info.git_model_id}"
                )
            changed_files, deleted_files = self._repo.find_changed_files(
                self.github_sha, from_commit_sha
            )
            logger.debug(
                "Model %s changes since commit %s. Changed files: %s. Deleted files: %s.",
                model_info.git_model_id,
                from_commit_sha,
                changed_files,
                deleted_files,
            )
            self._handle_changed_or_new_files(model_info, changed_files)
            self._handle_deleted_files(model_info, deleted_files)

    @staticmethod
    def _handle_changed_or_new_files(model_info, changed_or_new_files):
        for changed_file in changed_or_new_files:
            changed_model_filepath = model_info.model_file_paths.get(changed_file)
            if changed_model_filepath:
                logger.debug(
                    "Changed/new file '%s' affects model '%s'",
                    changed_file,
                    model_info.model_path.name,
                )
                model_info.file_changes.add_changed(changed_model_filepath)

            # A change could happen in the model's definition (yaml), which is not necessarily
            # included by the glob patterns
            if model_info.yaml_filepath.samefile(changed_file):
                model_info.flags.should_update_settings = True

    def _handle_deleted_files(self, model_info, deleted_files):
        for deleted_file in deleted_files:
            # Being stateless, check each deleted file against the stored custom model version
            # in DataRobot
            if model_info.git_model_id in self.datarobot_models:
                latest_version = self.datarobot_models[model_info.git_model_id].latest_version
                if latest_version:
                    model_root_dir = model_info.model_path
                    repo_root_dir = self.options.root_dir
                    path_under_model, _ = ModelFilePath.get_path_under_model(
                        deleted_file, model_root_dir, repo_root_dir
                    )
                    if not path_under_model:
                        raise UnexpectedResult(f"The path '{deleted_file}' is outside the repo.")

                    model_info.file_changes.extend_deleted(
                        [
                            item["id"]
                            for item in latest_version["items"]
                            if path_under_model == item["filePath"]
                        ]
                    )
                    logger.debug(
                        "File path %s will be deleted from model %s.",
                        deleted_file,
                        model_info.git_model_id,
                    )

    def _apply_datarobot_actions_for_affected_models(self):
        logger.info("Apply DataRobot actions for affected models ...")
        self._handle_model_changes()
        self._handle_deleted_models()

    def _handle_model_changes(self):
        for git_model_id, model_info in self.models_info.items():
            if model_info.is_affected_by_commit:
                logger.info("Model '%s' is affected by commit.", model_info.model_path)

                if model_info.should_create_new_version:
                    custom_model, already_existed = self._get_or_create_custom_model(model_info)
                    if not already_existed:
                        self._stats.total_created += 1

                    custom_model_id = custom_model["id"]
                    version_id = self._create_custom_model_version(custom_model_id, model_info)
                    if model_info.should_run_test:
                        self._test_custom_model_version(custom_model_id, version_id, model_info)

                    logger.info(
                        "Custom inference model version was successfully created. "
                        "git_model_id: %s, model_id: %s, version_id: %s.",
                        git_model_id,
                        custom_model_id,
                        version_id,
                    )
                else:
                    custom_model = self.datarobot_models[model_info.git_model_id].model

                if model_info.flags.should_update_settings:
                    self._update_settings(custom_model, model_info)

                self._stats.total_affected += 1

    def _get_or_create_custom_model(self, model_info):
        already_exists = model_info.git_model_id in self.datarobot_models
        if already_exists:
            custom_model = self.datarobot_models[model_info.git_model_id].model
        else:
            custom_model = self._dr_client.create_custom_model(model_info)
            logger.info("Custom inference model was created: %s", custom_model["id"])
        return custom_model, already_exists

    def _create_custom_model_version(self, custom_model_id, model_info):
        if model_info.flags.should_upload_all_files:
            changed_file_paths = list(model_info.model_file_paths.values())
        else:
            changed_file_paths = model_info.file_changes.changed_or_new_files

        logger.info(
            "Create custom inference model version. git_model_id:  %s, from_latest: %s",
            model_info.git_model_id,
            model_info.flags.should_upload_all_files,
        )
        logger.debug(
            "Files to be uploaded: %s, git_model_id: %s",
            [p.under_model for p in changed_file_paths],
            model_info.git_model_id,
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
            "GitHub version info. Ref name: %s, commit URL: %s, main branch commit sha: %s, "
            "pull request commit sha: %s",
            self.ref_name,
            commit_url,
            main_branch_commit_sha,
            pull_request_commit_sha,
        )

        return self._dr_client.create_custom_model_version(
            custom_model_id,
            model_info,
            self.ref_name,
            commit_url,
            main_branch_commit_sha,
            pull_request_commit_sha,
            changed_file_paths,
            model_info.file_changes.deleted_file_ids,
            from_latest=not model_info.flags.should_upload_all_files,
        )

    def _test_custom_model_version(self, model_id, model_version_id, model_info):
        logger.info("Executing custom model test ...")
        self._dr_client.run_custom_model_version_testing(model_id, model_version_id, model_info)

    def _update_settings(self, datarobot_custom_model, model_info):
        self._update_training_and_holdout_datasets(datarobot_custom_model, model_info)
        self._update_model_settings(datarobot_custom_model, model_info)

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
                    "Git model ID: %s. Training dataset name: %s. Training dataset ID: %s. "
                    "Holdout dataset name: %s. Holdout dataset ID: %s",
                    model_info.git_model_id,
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
                    "Training dataset was updated for structured model. Git model ID: %s. "
                    "Dataset name: %s. Dataset ID: %s. Dataset version ID: %s.",
                    model_info.git_model_id,
                    custom_model["trainingDataFileName"],
                    custom_model["trainingDatasetId"],
                    custom_model["trainingDatasetVersionId"],
                )

    def _update_model_settings(self, datarobot_custom_model, model_info):
        custom_model = self._dr_client.update_model_settings(datarobot_custom_model, model_info)
        if custom_model:
            logger.info("Model settings were updated. Git model ID: %s.", model_info.git_model_id)

    def _handle_deleted_models(self):
        missing_locally_id_to_git_id = {}
        for git_model_id, datarobot_model in self.datarobot_models.items():
            if git_model_id not in self.models_info:
                missing_locally_id_to_git_id[datarobot_model.model["id"]] = git_model_id

        if missing_locally_id_to_git_id:
            if not self.options.allow_model_deletion:
                missing_git_model_ids = list(missing_locally_id_to_git_id.values())
                raise IllegalModelDeletion(
                    "Model deletion was configured as not being allowed. "
                    f"The missing models in the local source tree are: {missing_git_model_ids}"
                )

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
            if any(model_id == deployment["customModel"]["id"] for deployment in deployments):
                logger.warning(
                    "Skipping model deletion because it is deployed. git_model_id: %s, "
                    "model_id: %s",
                    git_model_id,
                    model_id,
                )
                continue
            try:
                self._dr_client.delete_custom_model_by_model_id(model_id)
                self._stats.total_deleted += 1
                self._stats.total_affected += 1
                logger.info(
                    "Model was deleted with success. git_model_id: %s, model_id: %s",
                    git_model_id,
                    model_id,
                )
            except DataRobotClientError as ex:
                logger.error(str(ex))
