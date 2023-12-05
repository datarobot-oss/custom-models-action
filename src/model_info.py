#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A module that contains information about a model that was scanned and loaded from the local
source tree.
"""

import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import List

from model_file_path import ModelFilePath
from schema_validator import ModelSchema
from schema_validator import SharedSchema

logger = logging.getLogger()


# pylint: disable=too-many-public-methods
class InfoBase(ABC):
    """An abstract base class for models and deployments information classes."""

    @property
    @abstractmethod
    def schema_validator(self):
        """The schema validator glass."""

    @property
    @abstractmethod
    def yaml_filepath(self):
        """The yaml file path from which the given metadata definition is read from."""

    @property
    @abstractmethod
    def metadata(self):
        """The metadata dictionary."""

    @property
    @abstractmethod
    def user_provided_id(self):
        """
        A unique ID that is provided by the user to identify a given definition and read from
        the metadata.
        """

    def get_value(self, key, *sub_keys):
        """
        Get a value from the entity metadata given a key and sub-keys.

        Parameters
        ----------
        key : str
            A key name from the Schema.
        sub_keys :
            An optional dynamic sub-keys from the Schema.

        Returns
        -------
        Any or None,
            The value associated with the provided key (and sub-keys) or None if not exists.
        """

        return self.schema_validator.get_value(self.metadata, key, *sub_keys)

    def get_settings_value(self, key, *sub_keys):
        """
        Get a value from the metadata settings section, given a key and sub-keys under
        the settings section.

        Parameters
        ----------
        key : str
            A key name from the Schema, which is supposed to be under the
            SharedSchema.SETTINGS_SECTION_KEY section.
        sub_keys :
            An optional dynamic sub-keys from the ModelSchema, which are under the
            SharedSchema.SETTINGS_SECTION_KEY section.

        Returns
        -------
        Any or None,
            The value associated with the provided key (and sub-keys) or None if not exists.
        """
        return self.get_value(self.schema_validator.SETTINGS_SECTION_KEY, key, *sub_keys)

    def set_value(self, key, *sub_keys, value):
        """
        Set a value in an entity metadata.

        Parameters
        ----------
        key : str
            A key name of an entity schema.
        sub_keys : list
            An optional dynamic sub-keys from the entity schema.
        value : Any
            A value to set for the given key and optionally sub keys.

        Returns
        -------
        dict,
            The revised metadata after the value was set.
        """

        return self.schema_validator.set_value(self.metadata, key, *sub_keys, value=value)

    def set_settings_value(self, key, *sub_keys, value):
        """
        Set a value in the entity metadata settings section.

        Parameters
        ----------
        key : str
            A key from the SharedSchema.SETTINGS_SECTION_KEY of the given schema validator.
        sub_keys: list
            An optional dynamic sub-keys from the enity schema.
        value : Any
            A value to set.

        Returns
        -------
        dict,
            The revised metadata after the value was set.
        """

        return self.set_value(SharedSchema.SETTINGS_SECTION_KEY, key, *sub_keys, value=value)


class ModelInfo(InfoBase):
    """Holds information about a model from the local source tree."""

    _model_file_paths: Dict[Path, ModelFilePath]

    @dataclass
    class Flags:
        """Contains flags to indicate certain conditions."""

        should_upload_all_files: bool = False
        should_update_settings: bool = False

        @property
        def should_create_version_from_latest(self):
            """
            A property to return whether a new model version should be created from latest version.
            """

            return not self.should_upload_all_files

    @dataclass
    class FileChanges:
        """Contains lists of changed/new and deleted files."""

        changed_or_new_files: List[ModelFilePath] = field(default_factory=list)
        deleted_file_ids: List[str] = field(default_factory=list)

        def add_changed(self, model_file_path):
            """Add model file to the changes/new list."""

            self.changed_or_new_files.append(model_file_path)

        def add_deleted_file_id(self, deleted_file_id):
            """Add a file ID to the deleted file IDs list."""

            self.deleted_file_ids.append(deleted_file_id)

    def __init__(self, yaml_filepath, model_path, metadata):
        self._yaml_filepath = Path(yaml_filepath)
        self._model_path = Path(model_path)
        self._metadata = metadata
        self._model_file_paths = {}
        self.file_changes = self.FileChanges()
        self.flags = self.Flags()

    @property
    def schema_validator(self):
        """Return the schema validator class."""

        return ModelSchema

    @property
    def yaml_filepath(self):
        """The yaml file path from which the given model metadata definition was read from"""
        return self._yaml_filepath

    @property
    def model_path(self):
        """The model's root directory"""
        return self._model_path

    @property
    def metadata(self):
        """The model's metadata"""
        return self._metadata

    @property
    def user_provided_id(self):
        """A model's unique ID that is provided by the user and read from the model's metadata"""
        return self.metadata[ModelSchema.MODEL_ID_KEY]

    @property
    def model_file_paths(self):
        """A list of file paths that associated with the given model"""
        return self._model_file_paths

    @property
    def is_binary(self):
        """Whether the given model's target type is binary"""
        return ModelSchema.is_binary(self.metadata)

    @property
    def is_regression(self):
        """Whether the given model's target type is regression"""
        return ModelSchema.is_regression(self.metadata)

    @property
    def is_unstructured(self):
        """Whether the given model's target type is unstructured"""
        return ModelSchema.is_unstructured(self.metadata)

    @property
    def is_multiclass(self):
        """Whether the given model's target type is multi-class"""
        return ModelSchema.is_multiclass(self.metadata)

    def main_program_filepath(self):
        """Returns the main program file path of the given model"""
        try:
            return next(p for _, p in self.model_file_paths.items() if p.name == "custom.py")
        except StopIteration:
            return None

    def main_program_exists(self):
        """Returns whether the main program file path exists or not"""
        return self.main_program_filepath() is not None

    def set_model_paths(self, paths, workspace_path):
        """
        Builds a dictionary of the files belong to the given model. The key is a resolved
        file path of a given file and the value is a ModelFilePath of that same file.

        Parameters
        ----------
        paths : list
            A list of file paths associated with the given model.
        workspace_path : pathlib.Path
            The repository root directory.
        """

        logger.debug("Model %s is set with the following paths: %s", self.user_provided_id, paths)
        self._model_file_paths = {}
        for path in paths:
            model_filepath = ModelFilePath(path, self.model_path, workspace_path)
            self._model_file_paths[model_filepath.resolved] = model_filepath

    def paths_under_model_by_relative(self, relative_to):
        """
        Returns a list (as a set) of the model's files that subjected to specific relative value.

        Parameters
        ----------
        relative_to : ModelFilePath.RelativeTo
            The relation value.

        Returns
        -------
        set,
            The list of the model's that subjected to the given input relative value.
        """

        return set(
            p.under_model for _, p in self.model_file_paths.items() if p.relative_to == relative_to
        )

    def is_affected_by_commit(self, datarobot_latest_model_version):
        """Whether the given model is affected by the last commit"""

        return (
            self.flags.should_update_settings
            or self.should_create_new_version(datarobot_latest_model_version)
            or self.is_there_a_change_in_training_or_holdout_data_at_version_level(
                datarobot_latest_model_version
            )
        )

    def should_create_new_version(self, datarobot_latest_model_version):
        """Whether a new custom inference model version should be created"""

        if (
            not datarobot_latest_model_version
            or self.flags.should_upload_all_files
            or bool(self.file_changes.changed_or_new_files)
            or bool(self.file_changes.deleted_file_ids)
        ):
            logger.debug(
                "Need to create new version. datarobot_latest_model_version:%s "
                "should_upload_all_files:%s changed_or_new_files:%s deleted_file_ids:%s",
                datarobot_latest_model_version,
                self.flags.should_upload_all_files,
                self.file_changes.changed_or_new_files,
                self.file_changes.deleted_file_ids,
            )
            return True

        for resource_key, dr_attribute_key in (
            (ModelSchema.MEMORY_KEY, "maximumMemory"),
            (ModelSchema.REPLICAS_KEY, "replicas"),
            (ModelSchema.EGRESS_NETWORK_POLICY_KEY, "networkEgressPolicy"),
        ):
            configured_resource = self.get_value(ModelSchema.VERSION_KEY, resource_key)
            if configured_resource and configured_resource != datarobot_latest_model_version.get(
                dr_attribute_key
            ):
                logger.debug(
                    "Need to create new version. Resource '%s' changed. "
                    "Configured value: '%s' Value on server: '%s'",
                    resource_key,
                    configured_resource,
                    datarobot_latest_model_version.get(dr_attribute_key),
                )
                return True

        return False

    def is_there_a_change_in_training_or_holdout_data_at_version_level(
        self, datarobot_latest_model_version
    ):
        """Whether there's a model's version configuration change of a training/holdout data."""

        # Check training dataset
        configured_training_dataset = self.get_value(
            ModelSchema.VERSION_KEY, ModelSchema.TRAINING_DATASET_ID_KEY
        )
        if datarobot_latest_model_version is None:
            # This can happen only when the custom model is first created and still does not have
            # any associated version. A hidden assumption is that a holdout data will never be set
            # without a training data
            return configured_training_dataset is not None

        latest_training_dataset = datarobot_latest_model_version.get("training_data", {}).get(
            "dataset_id"
        )
        if configured_training_dataset != latest_training_dataset:
            logger.debug("Configured training dataset != latest training dataset")
            return True

        # Check holdout
        if self.is_unstructured:
            configured_holdout_dataset = self.get_value(
                ModelSchema.VERSION_KEY, ModelSchema.TRAINING_DATASET_ID_KEY
            )
            latest_holdout_dataset = datarobot_latest_model_version.get("holdout_data", {}).get(
                "dataset_id"
            )
            if configured_holdout_dataset != latest_holdout_dataset:
                logger.debug("Configured holdout dataset != latest holdout dataset")
                return True

        else:
            configured_holdout_partition = self.get_value(
                ModelSchema.VERSION_KEY, ModelSchema.PARTITIONING_COLUMN_KEY
            )
            latest_partition = datarobot_latest_model_version.get("holdout_data", {}).get(
                "partition_column"
            )
            if configured_holdout_partition != latest_partition:
                logger.debug("Configured holdout partition != latest holdout partition")
                return True

        return False

    @property
    def should_register_model(self):
        """Wheter this model should be added as a registered model."""
        return self.registered_model_name is not None

    @property
    def registered_model_name(self):
        """The registered model name to use or None if model should not be registered."""
        return self.get_value(ModelSchema.MODEL_REGISTRY_KEY, ModelSchema.MODEL_NAME)

    @property
    def registered_model_description(self):
        """The registered model description to use or None if model should not be registered."""
        return self.get_value(ModelSchema.MODEL_REGISTRY_KEY, ModelSchema.MODEL_DESCRIPTION)

    @property
    def registered_model_global(self):
        """Wheter the registered model should be global or not."""
        return self.get_value(ModelSchema.MODEL_REGISTRY_KEY, ModelSchema.GLOBAL)

    @property
    def should_run_test(self):
        """
        Querying the model's metadata and check whether a custom model testing should be executed.
        """
        return ModelSchema.TEST_KEY in self.metadata and not self.get_value(
            ModelSchema.TEST_KEY, ModelSchema.TEST_SKIP_KEY
        )
