import logging

from bson import ObjectId
from schema import And, Use
from schema import Schema
from schema import SchemaError
from schema import Optional
from schema import Or

from common.convertors import MemoryConvertor
from common.exceptions import InvalidModelSchema

logger = logging.getLogger()


class ModelSchema:
    MULTI_MODELS_KEY = "datarobot_models"
    MODEL_ENTRY_PATH_KEY = "model_path"
    MODEL_ENTRY_META_KEY = "model_metadata"
    MODEL_ID_KEY = "git_datarobot_model_id"

    TARGET_TYPE_KEY = "target_type"
    TARGET_TYPE_BINARY_KEY = "Binary"
    TARGET_TYPE_REGRESSION_KEY = "Regression"
    TARGET_TYPE_MULTICLASS_KEY = "Multiclass"
    TARGET_TYPE_ANOMALY_DETECTION_KEY = "Anomaly Detection"
    TARGET_TYPE_UNSTRUCTURED_BINARY_KEY = "Unstructured (Binary)"
    TARGET_TYPE_UNSTRUCTURED_REGRESSION_KEY = "Unstructured (Regression)"
    TARGET_TYPE_UNSTRUCTURED_MULTICLASS_KEY = "Unstructured (Multiclass)"
    TARGET_TYPE_UNSTRUCTURED_OTHER_KEY = "Unstructured (Other)"

    TARGET_NAME_KEY = "target_name"

    # Regression models
    PREDICTION_THRESHOLD_KEY = "prediction_threshold"

    # Binary models
    POSITIVE_CLASS_LABEL_KEY = "positive_class_label"
    NEGATIVE_CLASS_LABEL_KEY = "negative_class_label"

    # Multiclass models
    CLASS_LABELS_KEY = "class_labels"
    LANGUAGE_KEY = "language"

    SETTINGS_KEY = "settings"
    NAME_KEY = "name"
    DESCRIPTION_KEY = "description"
    TRAINING_DATASET_KEY = "training_dataset"
    HOLDOUT_DATASET_KEY = "holdout_dataset"

    DEPLOYMENT_ID_KEY = "git_datarobot_deployment_id"

    VERSION_KEY = "version"
    MODEL_ENV_KEY = "model_environment"
    INCLUDE_GLOB_KEY = "include_glob_pattern"
    EXCLUDE_GLOB_KEY = "exclude_glob_pattern"
    MEMORY_KEY = "memory"
    REPLICAS_KEY = "replicas"

    TEST_KEY = "test"
    TEST_DATA_KEY = "test_data"

    CHECKS_KEY = "checks"
    NULL_IMPUTATION_KEY = "null_imputation"
    CHECK_VALUE_KEY = "value"
    BLOCK_DEPLOYMENT_IF_FAILS_KEY = "block_deployment_if_fails"
    SIDE_EFFECT_KEY = "side_effect"
    PREDICTION_VERIFICATION_KEY = "prediction_verification"
    OUTPUT_DATASET_KEY = "output_dataset"
    MATCH_THRESHOLD_KEY = "match_threshold"
    PASSING_MATCH_RATE_KEY = "passing_match_rate"
    PERFORMANCE_KEY = "performance"
    MAXIMUM_RESPONSE_TIME_KEY = "maximum_response_time"
    CHECK_DURATION_LIMIT_KEY = "check_duration_limit"
    NUMBER_OF_PARALLEL_USERS_KEY = "number_of_parallel_users"
    STABILITY_KEY = "stability"
    TOTAL_PREDICTION_REQUESTS_KEY = "total_prediction_requests"
    PASSING_RATE_KEY = "passing_rate"
    NUMBER_OF_PARALLEL_USERS_KEY = "number_of_parallel_users"
    MINIMUM_PAYLOAD_SIZE_KEY = "minimum_payload_size"
    MAXIMUM_PAYLOAD_SIZE_KEY = "maximum_payload_size"

    MODEL_SCHEMA = Schema(
        {
            MODEL_ID_KEY: str,
            Optional(DEPLOYMENT_ID_KEY): str,
            TARGET_TYPE_KEY: Or(
                TARGET_TYPE_BINARY_KEY,
                TARGET_TYPE_REGRESSION_KEY,
                TARGET_TYPE_MULTICLASS_KEY,
                TARGET_TYPE_ANOMALY_DETECTION_KEY,
                TARGET_TYPE_UNSTRUCTURED_BINARY_KEY,
                TARGET_TYPE_UNSTRUCTURED_REGRESSION_KEY,
                TARGET_TYPE_UNSTRUCTURED_MULTICLASS_KEY,
                TARGET_TYPE_UNSTRUCTURED_OTHER_KEY,
            ),
            TARGET_NAME_KEY: str,
            Optional(PREDICTION_THRESHOLD_KEY): And(float, lambda n: 0 <= n <= 1),
            Optional(POSITIVE_CLASS_LABEL_KEY): str,
            Optional(NEGATIVE_CLASS_LABEL_KEY): str,
            Optional(CLASS_LABELS_KEY): list,
            Optional(LANGUAGE_KEY): str,
            Optional(SETTINGS_KEY): {
                Optional(NAME_KEY): str,
                Optional(DESCRIPTION_KEY): str,
                Optional(TRAINING_DATASET_KEY): And(str, lambda i: ObjectId.is_valid(i)),
                Optional(HOLDOUT_DATASET_KEY): And(str, lambda i: ObjectId.is_valid(i)),
            },
            VERSION_KEY: {
                MODEL_ENV_KEY: And(str, lambda i: ObjectId.is_valid(i)),
                Optional(INCLUDE_GLOB_KEY, default=[]): And(
                    list, lambda l: all(isinstance(e, str) for e in l)
                ),
                Optional(EXCLUDE_GLOB_KEY, default=[]): And(
                    list, lambda l: all(isinstance(x, str) for x in l)
                ),
                Optional(MEMORY_KEY): Use(lambda v: MemoryConvertor.to_bytes(v)),
                Optional(REPLICAS_KEY): And(int, lambda r: r > 0),
            },
            Optional(TEST_KEY): {
                TEST_DATA_KEY: And(str, lambda i: ObjectId.is_valid(i)),
                Optional(MEMORY_KEY): Use(lambda v: MemoryConvertor.to_bytes(v)),
                Optional(CHECKS_KEY): {
                    Optional(NULL_IMPUTATION_KEY): {
                        CHECK_VALUE_KEY: Or("yes", "no"),
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: Or("yes", "no"),
                    },
                    Optional(SIDE_EFFECT_KEY): {
                        CHECK_VALUE_KEY: Or("yes", "no"),
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: Or("yes", "no"),
                    },
                    Optional(PREDICTION_VERIFICATION_KEY): {
                        CHECK_VALUE_KEY: Or("yes", "no"),
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: Or("yes", "no"),
                        OUTPUT_DATASET_KEY: And(str, lambda i: ObjectId.is_valid(i)),
                        Optional(MATCH_THRESHOLD_KEY): And(float, lambda v: 0 <= v <= 1),
                        Optional(PASSING_MATCH_RATE_KEY): And(int, lambda v: 0 <= v <= 100),
                    },
                    Optional(PERFORMANCE_KEY): {
                        CHECK_VALUE_KEY: Or("yes", "no"),
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: Or("yes", "no"),
                        Optional(MAXIMUM_RESPONSE_TIME_KEY): And(int, lambda v: 1 <= v <= 1800),
                        Optional(CHECK_DURATION_LIMIT_KEY): And(int, lambda v: 1 <= v <= 1800),
                        Optional(NUMBER_OF_PARALLEL_USERS_KEY): And(int, lambda v: 1 <= v <= 4),
                    },
                    Optional(STABILITY_KEY): {
                        CHECK_VALUE_KEY: Or("yes", "no"),
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: Or("yes", "no"),
                        Optional(TOTAL_PREDICTION_REQUESTS_KEY): And(int, lambda v: v >= 1),
                        Optional(PASSING_RATE_KEY): And(int, lambda v: 0 <= v <= 100),
                        Optional(NUMBER_OF_PARALLEL_USERS_KEY): And(int, lambda v: 1 <= v <= 4),
                        Optional(MINIMUM_PAYLOAD_SIZE_KEY): And(int, lambda v: v >= 1),
                        Optional(MAXIMUM_PAYLOAD_SIZE_KEY): And(int, lambda v: v >= 1),
                    },
                },
            },
        }
    )
    MULTI_MODELS_SCHEMA = Schema(
        {MULTI_MODELS_KEY: [{MODEL_ENTRY_PATH_KEY: str, MODEL_ENTRY_META_KEY: MODEL_SCHEMA.schema}]}
    )

    @classmethod
    def is_single_model_schema(cls, metadata):
        """
        Checks whether the given metadata might be a model metadata

        Parameters
        ----------
        metadata : dict
            A model metadata

        Returns
        -------
        ,
            Whether the given metadata is suspected to be a model metadata
        """

        return cls.MODEL_ID_KEY in metadata

    @classmethod
    def is_multi_models_schema(cls, metadata):
        """
        Checks whether the given metadata is a multi-models schema

        Parameters
        ----------
        metadata : dict
            A multi-model metadata

        Returns
        -------
        bool,
            Whether the given metadata is suspected to be a multi-model metadata
        """

        return cls.MULTI_MODELS_KEY in metadata

    @classmethod
    def is_binary(cls, metadata):
        return metadata[ModelSchema.TARGET_TYPE_KEY] in [
            cls.TARGET_TYPE_BINARY_KEY,
            cls.TARGET_TYPE_UNSTRUCTURED_BINARY_KEY,
        ]

    @classmethod
    def is_regression(cls, metadata):
        return metadata[ModelSchema.TARGET_TYPE_KEY] in [
            cls.TARGET_TYPE_REGRESSION_KEY,
            cls.TARGET_TYPE_UNSTRUCTURED_REGRESSION_KEY,
        ]

    @classmethod
    def is_multiclass(cls, metadata):
        return metadata[ModelSchema.TARGET_TYPE_KEY] in [
            cls.TARGET_TYPE_MULTICLASS_KEY,
            cls.TARGET_TYPE_UNSTRUCTURED_MULTICLASS_KEY,
        ]

    @classmethod
    def validate_and_transform_single(cls, model_metadata):
        try:
            transformed = cls.MODEL_SCHEMA.validate(model_metadata)
            cls._validate_single_model(transformed)
            return transformed
        except SchemaError as se:
            raise InvalidModelSchema(se.code)

    @classmethod
    def _validate_single_model(cls, single_model_metadata):
        cls._validate_mutual_exclusive_keys(single_model_metadata)
        cls._validate_dependent_keys(single_model_metadata)
        logger.debug(
            f"Model configuration is valid (id: {single_model_metadata[cls.MODEL_ID_KEY]})."
        )

    @classmethod
    def validate_and_transform_multi(cls, multi_models_metadata):
        # Validates and transform
        try:
            transformed = cls.MULTI_MODELS_SCHEMA.validate(multi_models_metadata)
            for model_entry in transformed[cls.MULTI_MODELS_KEY]:
                cls._validate_single_model(model_entry[cls.MODEL_ENTRY_META_KEY])
            return transformed
        except SchemaError as se:
            raise InvalidModelSchema(se.code)

    @classmethod
    def _validate_mutual_exclusive_keys(cls, model_metadata):
        for binary_class_label_key in [cls.POSITIVE_CLASS_LABEL_KEY, cls.NEGATIVE_CLASS_LABEL_KEY]:
            mutual_exclusive_keys = {
                cls.PREDICTION_THRESHOLD_KEY,
                binary_class_label_key,
                cls.CLASS_LABELS_KEY,
            }
            if len(mutual_exclusive_keys & model_metadata.keys()) > 1:
                raise InvalidModelSchema(f"Only one of '{mutual_exclusive_keys}' keys is expected")

    @classmethod
    def _validate_dependent_keys(cls, model_metadata):
        if cls.is_binary(model_metadata):
            binary_label_keys = {
                ModelSchema.POSITIVE_CLASS_LABEL_KEY,
                ModelSchema.NEGATIVE_CLASS_LABEL_KEY,
            }
            if len(binary_label_keys & set(model_metadata.keys())) != 2:
                raise InvalidModelSchema(
                    f"Binary model must be defined with the '{binary_label_keys}' keys."
                )
        elif (
            cls.is_multiclass(model_metadata)
            and model_metadata.get(ModelSchema.CLASS_LABELS_KEY) is None
        ):
            raise InvalidModelSchema(
                f"Multiclass model must be define with the 'mapping_classes' key."
            )

        stability = (
            model_metadata.get(ModelSchema.TEST_KEY, {})
            .get(ModelSchema.CHECKS_KEY, {})
            .get(ModelSchema.STABILITY_KEY, {})
        )
        if stability:
            minimum_payload_size = stability.get(ModelSchema.MINIMUM_PAYLOAD_SIZE_KEY, 1)
            maximum_payload_size = stability.get(ModelSchema.MAXIMUM_PAYLOAD_SIZE_KEY, 1000)
            if maximum_payload_size < minimum_payload_size:
                raise InvalidModelSchema(
                    f"Stability test check minimum payload size ({minimum_payload_size}) "
                    f"is higher than the maximum ({maximum_payload_size})"
                )

    @classmethod
    def get_value(cls, metadata, *args):
        """
        Extract a value from the metadata, for a given key hierarchy. The assumption is that parent
        keys are always dictionaries.

        Parameters
        ----------
        metadata : dict
            A model schema dictionary.
        args : str
            A variable number of strings, representing key hierarchy in the metadata.

        Returns
        -------
            A value or None
        """

        value = metadata
        for index, arg in enumerate(args):
            if not isinstance(value, dict):
                value = None
                break

            value = value.get(arg)
        return value
