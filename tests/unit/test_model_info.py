#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=redefined-outer-name

"""Unit tests for ModelInfo."""

import pytest

from model_info import ModelInfo
from schema_validator import ModelSchema


@pytest.fixture
def model_info():
    """ModelInfo fixture."""
    model_schema = {
        ModelSchema.MODEL_ID_KEY: "abc123",
        ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_BINARY,
        ModelSchema.SETTINGS_SECTION_KEY: {
            ModelSchema.NAME_KEY: "Awesome Model",
            ModelSchema.TARGET_NAME_KEY: "target_column",
            ModelSchema.POSITIVE_CLASS_LABEL_KEY: "1",
            ModelSchema.NEGATIVE_CLASS_LABEL_KEY: "0",
        },
        ModelSchema.VERSION_KEY: {
            ModelSchema.MODEL_ENV_ID_KEY: "627790db5621558eedc4c7fa",
            ModelSchema.RUNTIME_PARAMETER_VALUES_KEY: [
                {
                    "name": "param",
                    "type": "numeric",
                    "value": 42,
                },
            ],
        },
    }

    metadata = ModelSchema.validate_and_transform_single(model_schema)
    return ModelInfo("", "", metadata)


@pytest.fixture
def model_info_with_credential(model_info):
    """ModelInfo with credential runtime parameter."""
    runtime_params = model_info.get_value(
        ModelSchema.VERSION_KEY, ModelSchema.RUNTIME_PARAMETER_VALUES_KEY
    )
    runtime_params.append(
        {
            ModelSchema.RUNTIME_PARAMETER_VALUE_NAME_KEY: "cred",
            ModelSchema.RUNTIME_PARAMETER_VALUE_TYPE_KEY: "credential",
            ModelSchema.RUNTIME_PARAMETER_VALUE_VALUE_KEY: "cred_name",
        }
    )
    model_info.set_value(
        ModelSchema.VERSION_KEY, ModelSchema.RUNTIME_PARAMETER_VALUES_KEY, value=runtime_params
    )
    return model_info


@pytest.fixture
def model_version():
    """ModelVersion fixture."""
    return {
        "baseEnvironmentId": "627790db5621558eedc4c7fa",
        "runtimeParameters": [{"fieldName": "param", "type": "numeric", "currentValue": 42}],
    }


def test_should_create_new_version(model_info, model_version):
    """Test that new version is created when runtime param has changed."""
    model_version["runtimeParameters"][0]["currentValue"] = 0

    assert model_info.should_create_new_version(datarobot_latest_model_version=model_version)


def test_should_not_create_new_version(model_info, model_version):
    """Test that new version is not created when runtime param is the same."""
    assert not model_info.should_create_new_version(datarobot_latest_model_version=model_version)


def test_fail_when_runtime_parameter_does_not_exist(model_info, model_version):
    """Test that an error is raised when runtime param does not exist."""
    model_version["runtimeParameters"] = []

    with pytest.raises(ValueError, match="Model version on server does not have runtime param:"):
        model_info.should_create_new_version(datarobot_latest_model_version=model_version)


def test_replace_runtime_param_credentials(model_info_with_credential):
    """Test replace runtime param credentials."""
    credentials = [{"name": "cred_name", "credentialId": "cred_id"}]

    model_info_with_credential = model_info_with_credential.replace_runtime_param_credentials(
        credentials
    )

    runtime_params = model_info_with_credential.get_value(
        ModelSchema.VERSION_KEY, ModelSchema.RUNTIME_PARAMETER_VALUES_KEY
    )
    assert runtime_params[1]["value"] == "cred_id"


def test_replace_runtime_param_credentials_not_found(model_info_with_credential):
    """Verify that error is raised when credential is not found."""
    credentials = [{"name": "not_in_model_info", "credentialId": "cred_id"}]

    with pytest.raises(ValueError, match="Failed to find credential with name"):
        model_info_with_credential.replace_runtime_param_credentials(credentials)
