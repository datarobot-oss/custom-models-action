#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A module that contains information about a deployment that was scanned and loaded from the local
source tree.
"""

from model_info import InfoBase
from schema_validator import DeploymentSchema
from schema_validator import SharedSchema


class DeploymentInfo(InfoBase):
    """Holds information about a deployment in the local source tree"""

    def __init__(self, yaml_path, deployment_metadata):
        self._yaml_path = yaml_path
        self._metadata = deployment_metadata

    @property
    def schema_validator(self):
        """Return the schema validator class."""

        return DeploymentSchema

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
