#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""Contains common constants."""

from enum import Enum

# Only once custom model type is supported
CUSTOM_MODEL_TYPE = "inference"


class Label(Enum):
    """An enum of entity labels."""

    MODELS = "models"
    DEPLOYMENTS = "deployments"


class CommitEffect(Enum):
    """An enum that list the effects/changes caused by a given commit."""

    MODEL_SETTINGS = "model-settings"
    MODEL_VERSION = "model-version"
    TRAINING_DATA_FOR_VERSIONS = "training-data-for-versions"
