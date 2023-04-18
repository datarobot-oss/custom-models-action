#  Copyright (c) 2023. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.


"""
A module to test the mapping between local schema attribute values to their corresponding attribute
values in DataRobot API.
"""
from schema import Optional

from dr_api_attrs import DrApiCustomModelChecks
from dr_api_attrs import DrApiModelSettings
from dr_api_attrs import DrApiTargetType
from schema_validator import ModelSchema


class TestDrApiAttrsMappings:
    """
    A class that contains unit-tests for attribute mappings between local schema and DataRboto
    public API.
    """

    def test_custom_model_checks_mapping(self):
        """
        A case to test model check attributes correlation between local and DataRobot API.
        """

        local_optional_checks = ModelSchema.MODEL_SCHEMA.schema[Optional(ModelSchema.TEST_KEY)][
            Optional(ModelSchema.CHECKS_KEY)
        ]
        for local_optional_check in local_optional_checks:
            assert DrApiCustomModelChecks.to_dr_attr(local_optional_check.schema)

    def test_model_settings_mapping(self):
        """
        A case to test model settings attributes correlation between local and DataRobot API.
        """

        local_setting_keys = ModelSchema.MODEL_SCHEMA.schema[ModelSchema.SETTINGS_SECTION_KEY]
        for local_setting_key in local_setting_keys:
            if isinstance(local_setting_key, Optional):
                assert DrApiModelSettings.to_dr_attr(local_setting_key.schema)
            else:
                assert DrApiModelSettings.to_dr_attr(local_setting_key)

    def test_target_type_mapping(self):
        """
        A case to test target type attributes correlation between local and DataRobot API.
        """

        local_target_types = ModelSchema.MODEL_SCHEMA.schema[ModelSchema.TARGET_TYPE_KEY].args
        for local_target_type in local_target_types:
            assert DrApiTargetType.to_dr_attr(local_target_type)
