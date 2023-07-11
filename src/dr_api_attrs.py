#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A module that contains constants and mappings between local and DataRobot public API attribute
names.
"""
from enum import Enum

from schema_validator import ModelSchema

# The 'inference' model type is the only supported one.
CUSTOM_MODEL_TYPE = "inference"


class DrApiCustomModelChecks:  # pylint: disable=too-few-public-methods
    """
    Contains mappings between local custom model checks (testing) and DataRobot attributes.
    """

    MAPPING = {
        ModelSchema.NULL_VALUE_IMPUTATION_KEY: "nullValueImputation",
        ModelSchema.SIDE_EFFECTS_KEY: "sideEffects",
        ModelSchema.PREDICTION_VERIFICATION_KEY: "predictionVerificationCheck",
        ModelSchema.PERFORMANCE_KEY: "performanceCheck",
        ModelSchema.STABILITY_KEY: "stabilityCheck",
    }

    @classmethod
    def to_dr_attr(cls, check_name):
        """
        A method to map between local attribute test name to DataRobot related test name.

        Parameters
        ----------
        check_name : str
            A local test name.

        Returns
        -------
            DataRobot related test name.

        """
        return cls.MAPPING[check_name]


class DrApiModelSettings:  # pylint: disable=too-few-public-methods
    """
    Contains mapping between local and DataRobot model settings.
    """

    class ReservedValues(Enum):
        """A placeholder for reserved values."""

        UNSET = 1

    MAPPING = {
        ModelSchema.NAME_KEY: "name",
        ModelSchema.DESCRIPTION_KEY: "description",
        ModelSchema.LANGUAGE_KEY: "language",
        ModelSchema.TARGET_NAME_KEY: "targetName",
        ModelSchema.PREDICTION_THRESHOLD_KEY: "predictionThreshold",
        ModelSchema.POSITIVE_CLASS_LABEL_KEY: "positiveClassLabel",
        ModelSchema.NEGATIVE_CLASS_LABEL_KEY: "negativeClassLabel",
        ModelSchema.CLASS_LABELS_KEY: "classLabels",
        # The following keys are dependent on whether the model is structured or unstructured
        # and therefore the actual mapping is done directly by accessing the
        # `STRUCTURED_TRAINING_HOLDOUT_MAPPING` & `UNSTRUCTURED_TRAINING_HOLDOUT_MAPPING` below.
        ModelSchema.PARTITIONING_COLUMN_KEY: ReservedValues.UNSET,
        ModelSchema.TRAINING_DATASET_ID_KEY: ReservedValues.UNSET,
        ModelSchema.HOLDOUT_DATASET_ID_KEY: ReservedValues.UNSET,
    }

    STRUCTURED_TRAINING_HOLDOUT_RESPONSE_MAPPING = {
        ModelSchema.TRAINING_DATASET_ID_KEY: "trainingDatasetId",
        ModelSchema.PARTITIONING_COLUMN_KEY: "trainingDataPartitionColumn",
    }

    STRUCTURED_TRAINING_HOLDOUT_PATCH_MAPPING = {
        ModelSchema.TRAINING_DATASET_ID_KEY: "datasetId",
        ModelSchema.PARTITIONING_COLUMN_KEY: "partitionColumn",
    }

    UNSTRUCTURED_TRAINING_HOLDOUT_MAPPING = {
        ModelSchema.TRAINING_DATASET_ID_KEY: "trainingDatasetId",
        ModelSchema.HOLDOUT_DATASET_ID_KEY: "holdoutDatasetId",
    }

    @classmethod
    def to_dr_attr(cls, local_schema_key):
        """
        Maps between local schema settings attribute to the corresponding attribute in DataRobot.
        """

        return cls.MAPPING[local_schema_key]


class DrApiTargetType:  # pylint: disable=too-few-public-methods
    """
    Contains mappings between local target types and DataRobot attributes.
    """

    MAPPING = {
        ModelSchema.TARGET_TYPE_BINARY: "Binary",
        ModelSchema.TARGET_TYPE_UNSTRUCTURED_BINARY: "Binary",
        ModelSchema.TARGET_TYPE_REGRESSION: "Regression",
        ModelSchema.TARGET_TYPE_UNSTRUCTURED_REGRESSION: "Regression",
        ModelSchema.TARGET_TYPE_MULTICLASS: "Multiclass",
        ModelSchema.TARGET_TYPE_UNSTRUCTURED_MULTICLASS: "Multiclass",
        ModelSchema.TARGET_TYPE_UNSTRUCTURED_OTHER: "Unstructured",
        ModelSchema.TARGET_TYPE_ANOMALY_DETECTION: "Anomaly",
        ModelSchema.TARGET_TYPE_TEXT_GENERATION: "TextGeneration",
    }

    @classmethod
    def to_dr_attr(cls, local_schema_key):
        """
        Maps between local schema settings attribute to the corresponding attribute in DataRobot.
        """

        return cls.MAPPING[local_schema_key]
