import uuid
from collections import namedtuple

import pytest
from bson import ObjectId

from exceptions import InvalidModelSchema
from itertools import combinations
from schema_validator import ModelSchema


class TestSchemaValidator:
    @staticmethod
    def create_partial_model_schema(is_single=True, num_models=1):
        def _partial_model_schema(model_name):
            return {
                ModelSchema.MODEL_ID_KEY: str(uuid.uuid4()),
                "target_name": "target_feature_col",
                "settings": {
                    "name": model_name,
                },
                "version": {"model_environment": str(ObjectId())},
            }

        if is_single:
            model_schema = _partial_model_schema(f"single-model")
        else:
            model_schema = {ModelSchema.MULTI_MODELS_KEY: []}
            for counter in range(1, num_models + 1):
                multi_models_key = ModelSchema.MULTI_MODELS_KEY
                model_schema[multi_models_key].append(
                    _partial_model_schema(f"model-{counter}")
                )
        return model_schema

    @pytest.fixture
    def regression_model_schema(self):
        return {
            ModelSchema.MODEL_ID_KEY: "abc123",
            "target_type": "Regression",
            "target_name": "target_column",
            "version": {
                "model_environment": "627785ea562155d227c6a56c",
            },
        }

    def test_is_single_models_schema(self):
        single_model_schema = self.create_partial_model_schema(is_single=True)
        assert ModelSchema.is_single_model_schema(single_model_schema)
        assert not ModelSchema.is_multi_models_schema(single_model_schema)

    def test_is_multi_models_schema(self):
        multi_model_schema = self.create_partial_model_schema(
            is_single=False, num_models=2
        )
        assert ModelSchema.is_multi_models_schema(multi_model_schema)
        assert not ModelSchema.is_single_model_schema(multi_model_schema)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_for_binary_model(self, is_single):
        def _set_binary_keys(schema):
            schema["target_type"] = "Binary"
            schema["positive_class_label"] = "1"
            schema["negative_class_label"] = "0"

        self._validate_for_model_type(is_single, _set_binary_keys)

    def _validate_for_model_type(self, is_single, setup_model_keys_func):
        model_schema = self.create_partial_model_schema(is_single, num_models=1)
        with pytest.raises(InvalidModelSchema):
            # Partial schema should fail
            self._validate_schema(is_single, model_schema)

        if is_single:
            setup_model_keys_func(model_schema)
        else:
            for single_model_schema in model_schema[ModelSchema.MULTI_MODELS_KEY]:
                setup_model_keys_func(single_model_schema)

        self._validate_schema(is_single, model_schema)

    @staticmethod
    def _validate_schema(is_single, model_schema):
        if is_single:
            ModelSchema().validate_and_transform_single(model_schema)
        else:
            ModelSchema().validate_and_transform_multi(model_schema)

    @staticmethod
    def _wrap_multi(model_schema):
        return {ModelSchema.MULTI_MODELS_KEY: [model_schema]}

    @pytest.mark.parametrize(
        "class_label_key", ["positive_class_label", "negative_class_label", None]
    )
    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_missing_key_for_binary_model(self, is_single, class_label_key):
        def _set_binary_keys(schema):
            schema["target_type"] = "Binary"
            if class_label_key:
                schema[class_label_key] = "fake"

        with pytest.raises(InvalidModelSchema):
            self._validate_for_model_type(is_single, _set_binary_keys)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_for_regression_model(self, is_single):
        def _set_regression_key(schema):
            schema["target_type"] = "Regression"

        self._validate_for_model_type(is_single, _set_regression_key)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_for_multiclass_model(self, is_single):
        def _set_multiclass_keys(schema):
            schema["target_type"] = "Multiclass"
            schema["mapping_classes"] = ["1", "2", "3"]

        self._validate_for_model_type(is_single, _set_multiclass_keys)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_missing_key_for_multiclass_model(self, is_single):
        def _set_multiclass_keys(schema):
            schema["target_type"] = "Multiclass"

        with pytest.raises(InvalidModelSchema):
            self._validate_for_model_type(is_single, _set_multiclass_keys)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    @pytest.mark.parametrize(
        "target_type",
        [
            "Unstructured (Binary)",
            "Unstructured (Regression)",
            "Unstructured (Multiclass)",
            "Unstructured (Other)",
        ],
    )
    def test_for_unstructured_model_types(self, is_single, target_type):
        def _set_unstructured_keys(schema):
            schema["target_type"] = target_type

        self._validate_for_model_type(is_single, _set_unstructured_keys)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_invalid_mutual_exclusive_keys(self, is_single):
        Key = namedtuple("Key", ["type", "name", "value"])
        mutual_exclusive_keys = {
            Key(type="Regression", name="prediction_threshold", value=0.5),
            (
                Key(type="Binary", name="positive_class_label", value="1"),
                Key(type="Binary", name="negative_class_label", value="0"),
            ),
            Key(type="Multiclass", name="mapping_classes", value=("a", "b", "c")),
        }

        def _set_single_model_keys(comb, schema):
            if not is_single:
                schema = schema[ModelSchema.MULTI_MODELS_KEY][0]
            for element in comb:
                if isinstance(element, Key):
                    # The 'type' is not really important here
                    schema["target_type"] = element.type
                    schema[element.name] = element.value
                else:
                    for key in element:
                        # The 'type' is not really important here
                        schema["target_type"] = key.type
                        schema[key.name] = key.value

        model_schema = self.create_partial_model_schema(is_single, num_models=1)
        comb_keys = combinations(mutual_exclusive_keys, 2)
        for comb in comb_keys:
            _set_single_model_keys(comb, model_schema)
            with pytest.raises(InvalidModelSchema):
                self._validate_schema(is_single, model_schema)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    @pytest.mark.parametrize(
        "mandatory_key_to_skip",
        [
            ModelSchema.MODEL_ID_KEY,
            "target_type",
            "target_name",
            "version",
            "version.model_environment",
        ],
    )
    def test_missing_mandatory_keys(
        self, is_single, mandatory_key_to_skip, regression_model_schema
    ):
        key_and_sub_key = mandatory_key_to_skip.split(".")
        key = key_and_sub_key[0]
        sub_key = key_and_sub_key[1] if len(key_and_sub_key) == 2 else None
        if sub_key:
            regression_model_schema[key].pop(sub_key)
        else:
            regression_model_schema.pop(key)

        if not is_single:
            regression_model_schema = self._wrap_multi(regression_model_schema)

        with pytest.raises(InvalidModelSchema) as e:
            self._validate_schema(is_single, regression_model_schema)

        assert f"Missing key: '{sub_key if sub_key else key}'" in str(e)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_full_model_schema(self, is_single):
        full_model_schema = {
            ModelSchema.MODEL_ID_KEY: "abc123",
            ModelSchema.DEPLOYMENT_ID_KEY: "edf456",
            "target_type": "Binary",
            "target_name": "target_column",
            "positive_class_label": "1",
            "negative_class_label": "0",
            "language": "Python",
            "settings": {
                "name": "Awesome Model",
                "description": "My awesome model",
                "training_dataset": "627790ba56215587b3021632",
                "holdout_dataset": "627790ca5621558b55c78d78",
            },
            "version": {
                "model_environment": "627790db5621558eedc4c7fa",
                "include_glob_pattern": ["./"],
                "exclude_glob_pattern": ["README.md", "out/"],
                "memory": "100Mi",
                "replicas": 3,
            },
            "test": {
                "test_data": "62779143562155aa34a3d65b",
                "memory": "100Mi",
                "checks": {
                    "null_imputation": {
                        "value": "yes",
                        "block_deployment_if_fails": "yes",
                    },
                    "side_effect": {
                        "value": "yes",
                        "block_deployment_if_fails": "yes",
                    },
                    "prediction_verification": {
                        "value": "yes",
                        "block_deployment_if_fails": "yes",
                    },
                    "prediction_verification": {
                        "value": "yes",
                        "block_deployment_if_fails": "no",
                        "output_dataset": "627791f5562155d63f367b05",
                        "match_threshold": 0.9,
                        "passing_match_rate": 85,
                    },
                    "performance": {
                        "value": "yes",
                        "block_deployment_if_fails": "no",
                        "maximum_response_time": 50,
                        "check_duration_limit": 100,
                        "number_of_parallel_users": 3,
                    },
                    "stability": {
                        "value": "no",
                        "block_deployment_if_fails": "yes",
                        "total_prediction_requests": 50,
                        "passing_rate": 95,
                        "number_of_parallel_users": 1,
                        "minimum_payload_size": 100,
                        "maximum_payload_size": 1000,
                    },
                },
            },
        }

        if not is_single:
            full_model_schema = self._wrap_multi(full_model_schema)

        self._validate_schema(is_single, full_model_schema)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_forbidden_extra_fields(self, is_single, regression_model_schema):
        forbidden_key = "forbidden_key"
        regression_model_schema[forbidden_key] = "Non allowed extra key"
        if not is_single:
            regression_model_schema = self._wrap_multi(regression_model_schema)
        with pytest.raises(InvalidModelSchema) as e:
            self._validate_schema(is_single, regression_model_schema)
        assert f"Wrong key '{forbidden_key}'" in str(e)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_dependent_stability_test_check_keys(
        self, is_single, regression_model_schema
    ):
        regression_model_schema["test"] = {
            "test_data": "62779bef562155562769f932",
            "checks": {
                "stability": {
                    "value": "yes",
                    "block_deployment_if_fails": "yes",
                    "minimum_payload_size": 100,
                    "maximum_payload_size": 50,
                }
            },
        }
        if not is_single:
            regression_model_schema = self._wrap_multi(regression_model_schema)
        with pytest.raises(InvalidModelSchema) as e:
            self._validate_schema(is_single, regression_model_schema)
        "Stability test check minimum payload size (100) is higher than the maximum (50)" in str(
            e
        )
