#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A module that contains constants and mappings between local and DataRobot public API attribute
names.
"""

from schema_validator import ModelSchema

# The 'inference' model type is the only supported one.
CUSTOM_MODEL_TYPE = "inference"


class DrApiAttrs:  # pylint: disable=too-few-public-methods
    """
    Contains mappings between local and DataRobot attributes.
    """

    DR_TEST_CHECK_MAP = {
        ModelSchema.NULL_VALUE_IMPUTATION_KEY: "nullValueImputation",
        ModelSchema.SIDE_EFFECTS_KEY: "sideEffects",
        ModelSchema.PREDICTION_VERIFICATION_KEY: "predictionVerificationCheck",
        ModelSchema.PERFORMANCE_KEY: "performanceCheck",
        ModelSchema.STABILITY_KEY: "stabilityCheck",
    }

    @classmethod
    def to_dr_test_check(cls, check_name):
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
        return cls.DR_TEST_CHECK_MAP[check_name]
