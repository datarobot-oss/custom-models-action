#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A module that contains information about a deployment that was scanned and loaded from the local
source tree.
"""

from schema_validator import DeploymentSchema
from schema_validator import SharedSchema


class DeploymentInfo:
    """Holds information about a deployment in the local source tree"""

    def __init__(self, yaml_path, deployment_metadata):
        self._yaml_path = yaml_path
        self._metadata = deployment_metadata

    @property
    def user_provided_id(self):
        """
        A deployment's unique ID that is provided by the user and read from the deployment's
        metadata
        """
        return self._metadata[DeploymentSchema.DEPLOYMENT_ID_KEY]

    @property
    def user_provided_model_id(self):
        """
        A model's unique ID that is provided by the user and read from the deployment's
        metadata. There should be a corresponding model's definition in the source tree,
        which corresponds to this model's unique ID.
        """
        return self._metadata[SharedSchema.MODEL_ID_KEY]

    @property
    def metadata(self):
        """The deployment's metadata"""
        return self._metadata

    @property
    def yaml_filepath(self):
        """The yaml file path from which the given deployment metadata definition was read from"""
        return self._yaml_path

    @property
    def is_challenger_enabled(self):
        """Whether a model challenger is enabled for the given deployment"""
        challenger_enabled = self.get_settings_value(DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY)
        # A special case, in which the default is to enable challengers
        return True if challenger_enabled is None else challenger_enabled

    def get_value(self, key, *sub_keys):
        """
        Get a value from the deployment's metadata given a key and sub-keys.

        Parameters
        ----------
        key : str
            A key name from the DeploymentSchema.
        sub_keys :
            An optional dynamic sub-keys from the DeploymentSchema.

        Returns
        -------
        Any or None,
            The value associated with the provided key (and sub-keys) or None if not exists.
        """

        return DeploymentSchema.get_value(self.metadata, key, *sub_keys)

    def set_value(self, key, *sub_keys, value):
        """
        Set a value in the deployment's metadata.

        Parameters
        ----------
        key : str
            A key name from the DeploymentSchema.
        sub_keys : list
            An optional dynamic sub-keys from the DeploymentSchema.
        value : Any
            A value to set for the given key and optionally sub keys.

        Returns
        -------
        dict,
            The revised metadata after the value was set.
        """

        return DeploymentSchema.set_value(self.metadata, key, *sub_keys, value=value)

    def get_settings_value(self, key, *sub_keys):
        """
        Get a value from the deployment's metadata settings section, given a key and sub-keys
        under the settings section.

        Parameters
        ----------
        key : str
            A key name from the DeploymentSchema, which is supposed to be under the
            SharedSchema.SETTINGS_SECTION_KEY section.
        sub_keys :
            An optional dynamic sub-keys from the DeploymentSchema, which are under the
            SharedSchema.SETTINGS_SECTION_KEY section.

        Returns
        -------
        Any or None,
            The value associated with the provided key (and sub-keys) or None if not exists.
        """

        return self.get_value(DeploymentSchema.SETTINGS_SECTION_KEY, key, *sub_keys)

    def set_settings_value(self, key, *sub_keys, value):
        """
        Set a value in the self deployment's metadata settings section.

        Parameters
        ----------
        key : str
            A key from the SharedSchema.SETTINGS_SECTION_KEY.
        sub_keys: list
            An optional dynamic sub-keys from the DeploymentSchema.
        value : Any
            A value to set.

        Returns
        -------
        dict,
            The revised metadata after the value was set.
        """

        return self.set_value(DeploymentSchema.SETTINGS_SECTION_KEY, key, *sub_keys, value=value)
