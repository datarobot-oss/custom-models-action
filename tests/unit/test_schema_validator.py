#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A module that contains unit-tests for schema validators."""

import copy
from collections import namedtuple
from itertools import combinations

import pytest
from bson import ObjectId

from common.exceptions import InvalidModelSchema
from common.exceptions import InvalidSchema
from common.exceptions import UnexpectedType
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from tests.unit.conftest import create_partial_deployment_schema
from tests.unit.conftest import create_partial_model_schema


class TestModelSchemaValidator:
    """Contains cases to test the model schema validator."""

    @pytest.fixture
    def regression_model_schema(self):
        """A fixture to return a Regression model schema."""

        return {
            ModelSchema.MODEL_ID_KEY: "abc123",
            ModelSchema.TARGET_TYPE_KEY: "Regression",
            ModelSchema.SETTINGS_SECTION_KEY: {
                ModelSchema.NAME_KEY: "My Awesome Model",
                ModelSchema.TARGET_NAME_KEY: "target_column",
            },
            ModelSchema.VERSION_KEY: {ModelSchema.MODEL_ENV_ID_KEY: "627785ea562155d227c6a56c"},
        }

    def test_is_single_models_schema(self):
        """A case to test whether a given schema is of a single model's schema pattern."""

        single_model_schema = create_partial_model_schema(is_single=True)
        assert ModelSchema.is_single_model_schema(single_model_schema)
        assert not ModelSchema.is_multi_models_schema(single_model_schema)
        assert not DeploymentSchema.is_single_deployment_schema(single_model_schema)
        assert not DeploymentSchema.is_multi_deployments_schema(single_model_schema)

    def test_is_multi_models_schema(self):
        """A case to test whether a given schema is of a multi-models' schema pattern."""

        multi_model_schema = create_partial_model_schema(is_single=False, num_models=2)
        assert ModelSchema.is_multi_models_schema(multi_model_schema)
        assert not ModelSchema.is_single_model_schema(multi_model_schema)

    @pytest.mark.parametrize(
        "binary_target_type",
        [ModelSchema.TARGET_TYPE_BINARY, ModelSchema.TARGET_TYPE_UNSTRUCTURED_BINARY],
    )
    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_for_binary_model(self, binary_target_type, is_single):
        """A case to test a Binary custom model schema."""

        def _set_binary_keys(schema):
            schema[ModelSchema.TARGET_TYPE_KEY] = binary_target_type
            schema[ModelSchema.SETTINGS_SECTION_KEY][ModelSchema.POSITIVE_CLASS_LABEL_KEY] = "1"
            schema[ModelSchema.SETTINGS_SECTION_KEY][ModelSchema.NEGATIVE_CLASS_LABEL_KEY] = "0"

        self._validate_for_model_type(is_single, _set_binary_keys)

    def _validate_for_model_type(self, is_single, setup_model_keys_func):
        model_schema = create_partial_model_schema(is_single, num_models=1)
        with pytest.raises(InvalidSchema):
            # Partial schema should fail
            self._validate_schema(is_single, model_schema)

        if is_single:
            setup_model_keys_func(model_schema)
        else:
            for model_entry in model_schema[ModelSchema.MULTI_MODELS_KEY]:
                setup_model_keys_func(model_entry[ModelSchema.MODEL_ENTRY_META_KEY])

        self._validate_schema(is_single, model_schema)

    @staticmethod
    def _validate_schema(is_single, model_schema):
        if is_single:
            transformed_schema = ModelSchema.validate_and_transform_single(model_schema)

            # Validate existence of default values
            for glob_key in [
                ModelSchema.INCLUDE_GLOB_KEY,
                ModelSchema.EXCLUDE_GLOB_KEY,
            ]:
                assert isinstance(transformed_schema[ModelSchema.VERSION_KEY][glob_key], list)
        else:
            transformed_schema = ModelSchema.validate_and_transform_multi(model_schema)

            # Validate existence of default values
            for model_entry in transformed_schema[ModelSchema.MULTI_MODELS_KEY]:
                for glob_key in [
                    ModelSchema.INCLUDE_GLOB_KEY,
                    ModelSchema.EXCLUDE_GLOB_KEY,
                ]:
                    model_metadata = model_entry[ModelSchema.MODEL_ENTRY_META_KEY]
                    assert isinstance(model_metadata[ModelSchema.VERSION_KEY][glob_key], list)

    @pytest.mark.parametrize(
        "binary_target_type",
        [ModelSchema.TARGET_TYPE_BINARY, ModelSchema.TARGET_TYPE_UNSTRUCTURED_BINARY],
    )
    @pytest.mark.parametrize(
        "class_label_key",
        [ModelSchema.POSITIVE_CLASS_LABEL_KEY, ModelSchema.NEGATIVE_CLASS_LABEL_KEY, None],
    )
    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_missing_key_for_binary_model(self, binary_target_type, class_label_key, is_single):
        """A case to test missing key in a Binary model schema."""

        def _set_binary_keys(schema):
            schema[ModelSchema.TARGET_TYPE_KEY] = binary_target_type
            if class_label_key:
                schema[ModelSchema.SETTINGS_SECTION_KEY][class_label_key] = "fake"

        with pytest.raises(InvalidModelSchema):
            self._validate_for_model_type(is_single, _set_binary_keys)

    @pytest.mark.parametrize(
        "regression_target_type",
        [
            ModelSchema.TARGET_TYPE_REGRESSION,
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_REGRESSION,
        ],
    )
    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_for_regression_model(self, regression_target_type, is_single):
        """A case to test a Regression model schema."""

        def _set_regression_key(schema):
            schema[ModelSchema.TARGET_TYPE_KEY] = regression_target_type

        self._validate_for_model_type(is_single, _set_regression_key)

    @pytest.mark.parametrize(
        "multiclass_target_type",
        [
            ModelSchema.TARGET_TYPE_MULTICLASS,
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_MULTICLASS,
        ],
    )
    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_for_multiclass_model(self, multiclass_target_type, is_single):
        """A case to test a Mult-Class model schema."""

        def _set_multiclass_keys(schema):
            schema[ModelSchema.TARGET_TYPE_KEY] = multiclass_target_type
            schema[ModelSchema.SETTINGS_SECTION_KEY][ModelSchema.CLASS_LABELS_KEY] = ["1", "2", "3"]

        self._validate_for_model_type(is_single, _set_multiclass_keys)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_missing_key_for_multiclass_model(self, is_single):
        """A case to test a missing key in a Multi-Class model schema."""

        def _set_multiclass_keys(schema):
            schema[ModelSchema.TARGET_TYPE_KEY] = ModelSchema.TARGET_TYPE_MULTICLASS

        with pytest.raises(InvalidModelSchema):
            self._validate_for_model_type(is_single, _set_multiclass_keys)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_for_unstructured_model_type(self, is_single):
        """A case to test an unstructured model schema."""

        def _set_unstructured_keys(schema):
            schema[ModelSchema.TARGET_TYPE_KEY] = ModelSchema.TARGET_TYPE_UNSTRUCTURED_OTHER

        self._validate_for_model_type(is_single, _set_unstructured_keys)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_invalid_mutual_exclusive_keys_target_types(self, is_single):
        """A case to test an invalid mutual exclusive keys related to target types."""

        Key = namedtuple("Key", ["type", "name", "value"])
        mutual_exclusive_keys = {
            Key(
                type=ModelSchema.TARGET_TYPE_REGRESSION,
                name=ModelSchema.PREDICTION_THRESHOLD_KEY,
                value=0.5,
            ),
            (
                Key(
                    type=ModelSchema.TARGET_TYPE_BINARY,
                    name=ModelSchema.POSITIVE_CLASS_LABEL_KEY,
                    value="1",
                ),
                Key(
                    type=ModelSchema.TARGET_TYPE_BINARY,
                    name=ModelSchema.NEGATIVE_CLASS_LABEL_KEY,
                    value="0",
                ),
            ),
            Key(
                type=ModelSchema.TARGET_TYPE_MULTICLASS,
                name=ModelSchema.CLASS_LABELS_KEY,
                value=("a", "b", "c"),
            ),
        }

        def _set_single_model_keys(comb, schema):
            if not is_single:
                schema = schema[ModelSchema.MULTI_MODELS_KEY][0][ModelSchema.MODEL_ENTRY_META_KEY]
            for element in comb:
                if isinstance(element, Key):
                    # The 'type' is not really important here
                    schema[ModelSchema.TARGET_TYPE_KEY] = element.type
                    schema[ModelSchema.SETTINGS_SECTION_KEY][element.name] = element.value
                else:
                    for key in element:
                        # The 'type' is not really important here
                        schema[ModelSchema.TARGET_TYPE_KEY] = key.type
                        schema[ModelSchema.SETTINGS_SECTION_KEY][key.name] = key.value

        model_schema = create_partial_model_schema(is_single, num_models=1)
        comb_keys = combinations(mutual_exclusive_keys, 2)
        for comb in comb_keys:
            _set_single_model_keys(comb, model_schema)
            with pytest.raises(InvalidSchema):
                self._validate_schema(is_single, model_schema)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    @pytest.mark.parametrize("section", [ModelSchema.SETTINGS_SECTION_KEY, ModelSchema.VERSION_KEY])
    def test_invalid_mutual_exclusive_keys_partitioning_and_holdout(self, is_single, section):
        """
        A case to test an invalid mutual exclusive keys related to partitioning and holdout
        attributes.
        """

        model_metadata = create_partial_model_schema(is_single, num_models=1, with_target_type=True)

        edit_metadata = (
            model_metadata
            if is_single
            else model_metadata[ModelSchema.MULTI_MODELS_KEY][0][ModelSchema.MODEL_ENTRY_META_KEY]
        )
        edit_metadata[section][ModelSchema.PARTITIONING_COLUMN_KEY] = "partitioning"
        edit_metadata[section][ModelSchema.HOLDOUT_DATASET_ID_KEY] = str(ObjectId())

        with pytest.raises(InvalidModelSchema):
            if is_single:
                ModelSchema.validate_and_transform_single(model_metadata)
            else:
                ModelSchema.validate_and_transform_multi(model_metadata)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    @pytest.mark.parametrize("is_unstructured", [True, False], ids=["unstructured", "structured"])
    @pytest.mark.parametrize("with_holdout", [True, False], ids=["with-holdout", "without-holdout"])
    def test_invalid_mutual_exclusive_training_and_holdout_keys_between_settings_and_version(
        self, is_single, is_unstructured, with_holdout
    ):
        """
        A case to test an invalid mutual exclusive keys between model settings section and version
        section, related to training and holdout dataset attributes.
        """

        model_metadata = create_partial_model_schema(is_single, num_models=1, with_target_type=True)

        edit_metadata = (
            model_metadata
            if is_single
            else model_metadata[ModelSchema.MULTI_MODELS_KEY][0][ModelSchema.MODEL_ENTRY_META_KEY]
        )
        sections_in_test = [ModelSchema.SETTINGS_SECTION_KEY, ModelSchema.VERSION_KEY]
        for index in range(2):
            section_one = sections_in_test[index]
            section_two = sections_in_test[index ^ 1]
            if is_unstructured:
                edit_metadata[section_one][ModelSchema.TRAINING_DATASET_ID_KEY] = str(ObjectId())
                edit_metadata[section_two][ModelSchema.TRAINING_DATASET_ID_KEY] = str(ObjectId())
                if with_holdout:
                    edit_metadata[section_one][ModelSchema.HOLDOUT_DATASET_ID_KEY] = str(ObjectId())
                    edit_metadata[section_two][ModelSchema.HOLDOUT_DATASET_ID_KEY] = str(ObjectId())
            else:
                edit_metadata[section_one][ModelSchema.TRAINING_DATASET_ID_KEY] = str(ObjectId())
                edit_metadata[section_two][ModelSchema.TRAINING_DATASET_ID_KEY] = str(ObjectId())
                if with_holdout:
                    edit_metadata[section_one][ModelSchema.PARTITIONING_COLUMN_KEY] = "partitioning"
                    edit_metadata[section_two][ModelSchema.PARTITIONING_COLUMN_KEY] = "partitioning"

            with pytest.raises(InvalidModelSchema):
                if is_single:
                    ModelSchema.validate_and_transform_single(model_metadata)
                else:
                    ModelSchema.validate_and_transform_multi(model_metadata)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    @pytest.mark.parametrize(
        "mandatory_key_to_skip",
        [
            ModelSchema.MODEL_ID_KEY,
            ModelSchema.TARGET_TYPE_KEY,
            ModelSchema.VERSION_KEY,
            ModelSchema.SETTINGS_SECTION_KEY,
            f"{ModelSchema.SETTINGS_SECTION_KEY}.{ModelSchema.NAME_KEY}",
            f"{ModelSchema.SETTINGS_SECTION_KEY}.{ModelSchema.TARGET_NAME_KEY}",
            f"{ModelSchema.VERSION_KEY}.{ModelSchema.MODEL_ENV_ID_KEY}",
        ],
    )
    def test_missing_mandatory_keys(
        self, is_single, mandatory_key_to_skip, regression_model_schema
    ):
        """A case to test missing mandatory keys in a schema."""

        key_and_sub_key = mandatory_key_to_skip.split(".")
        key = key_and_sub_key[0]
        sub_key = key_and_sub_key[1] if len(key_and_sub_key) == 2 else None
        if sub_key:
            regression_model_schema[key].pop(sub_key)
        else:
            regression_model_schema.pop(key)

        if not is_single:
            regression_model_schema = self._wrap_multi(regression_model_schema)

        with pytest.raises(InvalidSchema) as ex:
            self._validate_schema(is_single, regression_model_schema)

        assert f"Missing key: '{sub_key or key}'" in str(ex)

    @staticmethod
    def _wrap_multi(model_schema):
        return {
            ModelSchema.MULTI_MODELS_KEY: [
                {
                    ModelSchema.MODEL_ENTRY_PATH_KEY: "./m1",
                    ModelSchema.MODEL_ENTRY_META_KEY: model_schema,
                }
            ]
        }

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_full_model_schema(self, is_single, mock_full_binary_model_schema):
        """A case to test a full model Schema."""

        full_model_schema = mock_full_binary_model_schema.copy()
        if not is_single:
            full_model_schema = self._wrap_multi(full_model_schema)

        self._validate_schema(is_single, full_model_schema)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_forbidden_extra_fields(self, is_single, regression_model_schema):
        """A case to test forbidden extra fields."""

        forbidden_key = "forbidden_key"
        regression_model_schema[forbidden_key] = "Non allowed extra key"
        if not is_single:
            regression_model_schema = self._wrap_multi(regression_model_schema)
        with pytest.raises(InvalidSchema) as ex:
            self._validate_schema(is_single, regression_model_schema)
        assert f"Wrong key '{forbidden_key}'" in str(ex)

    @pytest.mark.parametrize("is_single", [True, False], ids=["single", "multi"])
    def test_dependent_stability_test_check_keys(self, is_single, regression_model_schema):
        """A case to test dependent keys in a custom model stability test check."""

        regression_model_schema[ModelSchema.TEST_KEY] = {
            ModelSchema.TEST_DATA_ID_KEY: "62779bef562155562769f932",
            ModelSchema.CHECKS_KEY: {
                ModelSchema.STABILITY_KEY: {
                    ModelSchema.CHECK_ENABLED_KEY: True,
                    ModelSchema.BLOCK_DEPLOYMENT_IF_FAILS_KEY: True,
                    ModelSchema.MINIMUM_PAYLOAD_SIZE_KEY: 100,
                    ModelSchema.MAXIMUM_PAYLOAD_SIZE_KEY: 50,
                }
            },
        }
        if not is_single:
            regression_model_schema = self._wrap_multi(regression_model_schema)
        with pytest.raises(InvalidModelSchema) as ex:
            self._validate_schema(is_single, regression_model_schema)
        assert (
            "Stability test check minimum payload size (100) is higher than the maximum (50)"
            in str(ex)
        )


class TestModelSchemaGetValue:
    """Contains cases to test the get value method from a model schema."""

    def test_first_level_key(self, mock_full_binary_model_schema):
        """A case to test a value reading from the top level of a schema."""

        input_metadata = mock_full_binary_model_schema
        returned_value = ModelSchema.get_value(input_metadata, ModelSchema.SETTINGS_SECTION_KEY)
        assert returned_value == input_metadata[ModelSchema.SETTINGS_SECTION_KEY]

    def test_second_level_key(self, mock_full_binary_model_schema):
        """A case to test a value reading from the second level of a schema."""

        input_metadata = mock_full_binary_model_schema
        returned_value = ModelSchema.get_value(
            input_metadata, ModelSchema.SETTINGS_SECTION_KEY, ModelSchema.DESCRIPTION_KEY
        )
        assert (
            returned_value
            == input_metadata[ModelSchema.SETTINGS_SECTION_KEY][ModelSchema.DESCRIPTION_KEY]
        )

    def test_non_existing_key_at_first_level(self, mock_full_binary_model_schema):
        """A case to test a value reading from non-existing key in the top level of a schema."""

        input_metadata = mock_full_binary_model_schema
        returned_value = ModelSchema.get_value(input_metadata, "non-existing-key")
        assert returned_value is None

    def test_non_existing_key_at_second_level(self, mock_full_binary_model_schema):
        """A case to test a value reading from non-existing key in the second level of a schema."""

        input_metadata = mock_full_binary_model_schema
        returned_value = ModelSchema.get_value(
            input_metadata, ModelSchema.SETTINGS_SECTION_KEY, "non-existing-key"
        )
        assert returned_value is None

    def test_non_existing_key_hierarchy(self, mock_full_binary_model_schema):
        """A case to test a value reading of non-exiting keys hierarchy in a schema."""

        input_metadata = mock_full_binary_model_schema
        returned_value = ModelSchema.get_value(
            input_metadata, ModelSchema.TARGET_TYPE_KEY, ModelSchema.MODEL_ID_KEY
        )
        assert returned_value is None

    def test_invalid_metadata_argument(self):
        """A case to test an invalid call with a wrong type of the first argument."""

        with pytest.raises(UnexpectedType):
            ModelSchema.get_value(ModelSchema.SETTINGS_SECTION_KEY, ModelSchema.NAME_KEY)


class TestModelSchemaSetValue:
    """Contains cases to test the set value method from a model schema."""

    @pytest.mark.parametrize("key_name", [ModelSchema.TARGET_TYPE_KEY, "non-existing-key"])
    def test_first_level_key(self, mock_full_binary_model_schema, key_name):
        """A case to test a value writing to the top level of a schema."""

        input_metadata = copy.deepcopy(mock_full_binary_model_schema)
        name = str(ObjectId())
        metadata = ModelSchema.set_value(input_metadata, key_name, value=name)
        assert metadata[key_name] == name
        assert input_metadata[key_name] == name

    @pytest.mark.parametrize(
        "section_name, key_name",
        [
            (ModelSchema.SETTINGS_SECTION_KEY, ModelSchema.NAME_KEY),
            ("non-existing-section", ModelSchema.NAME_KEY),
            (ModelSchema.SETTINGS_SECTION_KEY, "non-existing-key"),
            ("non-existing-section", "non-existing-key"),
        ],
        ids=[
            "existing-section-existing-key",
            "non-existing-section-existing-key",
            "existing-section-non-existing-key",
            "non-existing-section-non-existing-key",
        ],
    )
    def test_second_level_key(self, mock_full_binary_model_schema, section_name, key_name):
        """A case to test a value writing to the second level of a schema."""

        input_metadata = copy.deepcopy(mock_full_binary_model_schema)
        value = str(ObjectId())
        metadata = ModelSchema.set_value(input_metadata, section_name, key_name, value=value)
        assert metadata[section_name][key_name] == value
        assert input_metadata[section_name][key_name] == value

    def test_second_level_key_failure_due_to_non_dict_top_level(
        self, mock_full_binary_model_schema
    ):
        """A case to test an invalid call with an unexpected attribute type at the top level."""

        input_metadata = copy.deepcopy(mock_full_binary_model_schema)
        value = str(ObjectId())
        section_name = ModelSchema.TARGET_TYPE_KEY
        key_name = ModelSchema.NAME_KEY
        with pytest.raises(UnexpectedType):
            ModelSchema.set_value(input_metadata, section_name, key_name, value=value)

    def test_target_type_value_definitions_are_all_included(self):
        """
        A case to test that all the target type value definitions are included in the target type
        attribute in the model's schema.
        """

        target_type_value_definitions = {
            v
            for k, v in ModelSchema.__dict__.items()
            if k.startswith("TARGET_TYPE_") and not k.endswith("_KEY")
        }
        target_type_choices = set(ModelSchema.MODEL_SCHEMA.schema["target_type"].args)
        assert target_type_value_definitions == target_type_choices


class TestDeploymentSchemaValidator:
    """Contains cases to test a deployment schema validator."""

    def test_is_single_deployment_schema(self):
        """A case to test whether a given schema is of a single deployment's schema pattern."""

        single_deployment_schema = create_partial_deployment_schema(is_single=True)
        assert DeploymentSchema.is_single_deployment_schema(single_deployment_schema)
        assert not DeploymentSchema.is_multi_deployments_schema(single_deployment_schema)
        assert not ModelSchema.is_single_model_schema(single_deployment_schema)
        assert not ModelSchema.is_multi_models_schema(single_deployment_schema)

    def test_is_multi_models_schema(self):
        """A case to test whether a given schema is of a multi-deployments' schema pattern."""

        multi_deployments_schema = create_partial_deployment_schema(
            is_single=False, num_deployments=2
        )
        assert DeploymentSchema.is_multi_deployments_schema(multi_deployments_schema)
        assert not DeploymentSchema.is_single_deployment_schema(multi_deployments_schema)
        assert not ModelSchema.is_multi_models_schema(multi_deployments_schema)
        assert not ModelSchema.is_single_model_schema(multi_deployments_schema)
