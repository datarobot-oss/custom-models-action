import pytest

from model_info import ModelInfo
from schema_validator import ModelSchema


@pytest.fixture
def model_info():
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
                }
            ],
        },
    }

    metadata = ModelSchema.validate_and_transform_single(model_schema)
    return ModelInfo("", "", metadata)


@pytest.fixture
def model_version():
    return {
        "baseEnvironmentId": "627790db5621558eedc4c7fa",
        "runtimeParameters": [{"fieldName": "param", "type": "numeric", "currentValue": 42}],
    }


def test_should_create_new_version(model_info, model_version):
    model_version["runtimeParameters"][0]["currentValue"] = 0

    assert model_info.should_create_new_version(datarobot_latest_model_version=model_version)


def test_should_not_create_new_version(model_info, model_version):
    assert not model_info.should_create_new_version(datarobot_latest_model_version=model_version)


def test_fail_when_runtime_parameter_does_not_exist(model_info, model_version):
    model_version["runtimeParameters"] = []

    with pytest.raises(ValueError, match="Model version on server does not have runtime param:"):
        model_info.should_create_new_version(datarobot_latest_model_version=model_version)
