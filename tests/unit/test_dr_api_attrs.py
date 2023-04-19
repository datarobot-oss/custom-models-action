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
                local_name = local_setting_key.schema
            else:
                local_name = local_setting_key

            remote_key = DrApiModelSettings.to_dr_attr(local_name)
            if remote_key == DrApiModelSettings.ReservedValues.UNSET:
                remote_key = DrApiModelSettings.STRUCTURED_TRAINING_HOLDOUT_PATCH_MAPPING.get(
                    local_name
                )
                if not remote_key:
                    remote_key = DrApiModelSettings.UNSTRUCTURED_TRAINING_HOLDOUT_MAPPING.get(
                        local_name
                    )
                    assert remote_key, f"Missing local '{local_name}' key mapping!"

    def test_dr_structured_model_training_holdout_response_and_patch_keys(self):
        """
        Test the correlation between response and patch payloads of the training/holdout data in
        structured models.
        """

        response_mapping = DrApiModelSettings.STRUCTURED_TRAINING_HOLDOUT_RESPONSE_MAPPING
        patch_mapping = DrApiModelSettings.STRUCTURED_TRAINING_HOLDOUT_PATCH_MAPPING
        for local_key, _ in response_mapping.items():
            assert patch_mapping.get(local_key) is not None

    def test_target_type_mapping(self):
        """
        A case to test target type attributes correlation between local and DataRobot API.
        """

        local_target_types = ModelSchema.MODEL_SCHEMA.schema[ModelSchema.TARGET_TYPE_KEY].args
        for local_target_type in local_target_types:
            assert DrApiTargetType.to_dr_attr(local_target_type)
