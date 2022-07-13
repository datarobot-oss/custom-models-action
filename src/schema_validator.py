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


class SharedSchema:
    MODEL_ID_KEY = "git_datarobot_model_id"
    SETTINGS_SECTION_KEY = "settings"

    @classmethod
    def _validate_and_transform_single(cls, schema, metadata):
        try:
            transformed = schema.validate(metadata)
            cls._validate_single_transformed(transformed)
            return transformed
        except SchemaError as se:
            raise InvalidModelSchema(se.code)

    @classmethod
    def _validate_single_transformed(cls, single_transformed_metadata):
        cls._validate_mutual_exclusive_keys(single_transformed_metadata)
        cls._validate_dependent_keys(single_transformed_metadata)
        cls._validate_data_integrity(single_transformed_metadata)

    @classmethod
    def _validate_and_transform_multi(cls, schema, multi_metadata):
        # Validates and transform
        try:
            transformed = schema.validate(multi_metadata)
            for single_metadata in cls._next_single_transformed(transformed):
                cls._validate_single_transformed(single_metadata)
            return transformed
        except SchemaError as se:
            raise InvalidModelSchema(se.code)

    @classmethod
    def _next_single_transformed(cls, transformed_metadata):
        """
        A generator that must be implemented by a derived class. It returns the next
        metadata entry in a multi metadata definition.

        Parameters
        ----------
        transformed_metadata : dict
            A multi metadata structure.


        Returns
        -------
        metadata : dict
            A metadata that represents a single entity.
        """
        raise NotImplementedError("The '_next_metadata' must be implemented by a derived class.")

    @classmethod
    def _validate_mutual_exclusive_keys(cls, single_transformed_metadata):
        """
        Validates mutual exclusive keys in a single transformed metadata.
        Expected to be implemented by inherited class.

        Parameters
        ----------
        single_transformed_metadata : dict
            A metadata representation of a single entity after validation and transformation.
        """
        pass

    @classmethod
    def _validate_dependent_keys(cls, single_transformed_metadata):
        """
        Validates dependent keys in a single transformed metadata.
        Expected to be implemented by inherited class.

        Parameters
        ----------
        single_transformed_metadata : dict
            A metadata representation of a single entity after validation and transformation.
        """
        pass

    @classmethod
    def _validate_data_integrity(cls, single_transformed_metadata):
        """
        Validates data integrity in a single transformed metadata.
        Expected to be implemented by inherited class.

        Parameters
        ----------
        single_transformed_metadata : dict
            A metadata representation of a single entity after validation and transformation.
        """
        pass

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


class ModelSchema(SharedSchema):
    MULTI_MODELS_KEY = "datarobot_models"
    MODEL_ENTRY_PATH_KEY = "model_path"
    MODEL_ENTRY_META_KEY = "model_metadata"

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

    NAME_KEY = "name"
    DESCRIPTION_KEY = "description"
    TRAINING_DATASET_KEY = "training_dataset"
    HOLDOUT_DATASET_KEY = "holdout_dataset"

    VERSION_KEY = "version"
    MODEL_ENV_KEY = "model_environment"
    INCLUDE_GLOB_KEY = "include_glob_pattern"
    EXCLUDE_GLOB_KEY = "exclude_glob_pattern"
    MEMORY_KEY = "memory"
    REPLICAS_KEY = "replicas"

    TEST_KEY = "test"
    TEST_SKIP_KEY = "skip"
    TEST_DATA_KEY = "test_data"

    CHECKS_KEY = "checks"
    NULL_VALUE_IMPUTATION_KEY = "null_value_imputation"
    CHECK_ENABLED_KEY = "enabled"
    BLOCK_DEPLOYMENT_IF_FAILS_KEY = "block_deployment_if_fails"
    SIDE_EFFECTS_KEY = "side_effects"
    PREDICTION_VERIFICATION_KEY = "prediction_verification"
    OUTPUT_DATASET_KEY = "output_dataset"
    PREDICTIONS_COLUMN = "predictions_column"
    MATCH_THRESHOLD_KEY = "match_threshold"
    PASSING_MATCH_RATE_KEY = "passing_match_rate"
    PERFORMANCE_KEY = "performance"
    MAXIMUM_RESPONSE_TIME_KEY = "maximum_response_time"
    MAXIMUM_EXECUTION_TIME = "max_execution_time"
    NUMBER_OF_PARALLEL_USERS_KEY = "number_of_parallel_users"
    STABILITY_KEY = "stability"
    TOTAL_PREDICTION_REQUESTS_KEY = "total_prediction_requests"
    PASSING_RATE_KEY = "passing_rate"
    NUMBER_OF_PARALLEL_USERS_KEY = "number_of_parallel_users"
    MINIMUM_PAYLOAD_SIZE_KEY = "minimum_payload_size"
    MAXIMUM_PAYLOAD_SIZE_KEY = "maximum_payload_size"

    MODEL_SCHEMA = Schema(
        {
            SharedSchema.MODEL_ID_KEY: And(str, len),
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
            TARGET_NAME_KEY: And(str, len),
            Optional(PREDICTION_THRESHOLD_KEY): And(float, lambda n: 0 <= n <= 1),
            Optional(POSITIVE_CLASS_LABEL_KEY): And(str, len),
            Optional(NEGATIVE_CLASS_LABEL_KEY): And(str, len),
            Optional(CLASS_LABELS_KEY): list,
            Optional(LANGUAGE_KEY): And(str, len),
            SharedSchema.SETTINGS_SECTION_KEY: {
                NAME_KEY: And(str, len),
                Optional(DESCRIPTION_KEY): And(str, len),
                Optional(TRAINING_DATASET_KEY): And(str, lambda i: ObjectId.is_valid(i)),
                Optional(HOLDOUT_DATASET_KEY): And(str, lambda i: ObjectId.is_valid(i)),
            },
            VERSION_KEY: {
                MODEL_ENV_KEY: And(str, lambda i: ObjectId.is_valid(i)),
                Optional(INCLUDE_GLOB_KEY, default=[]): And(
                    list, lambda l: all(isinstance(e, str) and len(e) > 0 for e in l)
                ),
                Optional(EXCLUDE_GLOB_KEY, default=[]): And(
                    list, lambda l: all(isinstance(x, str) and len(x) > 0 for x in l)
                ),
                Optional(MEMORY_KEY): Use(lambda v: MemoryConvertor.to_bytes(v)),
                Optional(REPLICAS_KEY): And(int, lambda r: r > 0),
            },
            Optional(TEST_KEY): {
                # The skip attribute allows users to have the test section in their yaml file
                # and still disable testing
                Optional(TEST_SKIP_KEY, default=False): bool,
                Optional(TEST_DATA_KEY): And(str, lambda i: ObjectId.is_valid(i)),
                Optional(MEMORY_KEY): Use(lambda v: MemoryConvertor.to_bytes(v)),
                Optional(CHECKS_KEY): {
                    Optional(NULL_VALUE_IMPUTATION_KEY): {
                        CHECK_ENABLED_KEY: bool,
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: bool,
                    },
                    Optional(SIDE_EFFECTS_KEY): {
                        CHECK_ENABLED_KEY: bool,
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: bool,
                    },
                    Optional(PREDICTION_VERIFICATION_KEY): {
                        CHECK_ENABLED_KEY: bool,
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: bool,
                        OUTPUT_DATASET_KEY: And(str, lambda i: ObjectId.is_valid(i)),
                        PREDICTIONS_COLUMN: And(str, len),
                        Optional(MATCH_THRESHOLD_KEY): And(float, lambda v: 0 <= v <= 1),
                        Optional(PASSING_MATCH_RATE_KEY): And(int, lambda v: 0 <= v <= 100),
                    },
                    Optional(PERFORMANCE_KEY): {
                        CHECK_ENABLED_KEY: bool,
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: bool,
                        Optional(MAXIMUM_RESPONSE_TIME_KEY): And(int, lambda v: 1 <= v <= 1800),
                        Optional(MAXIMUM_EXECUTION_TIME): And(int, lambda v: 1 <= v <= 1800),
                        Optional(NUMBER_OF_PARALLEL_USERS_KEY): And(int, lambda v: 1 <= v <= 4),
                    },
                    Optional(STABILITY_KEY): {
                        CHECK_ENABLED_KEY: bool,
                        BLOCK_DEPLOYMENT_IF_FAILS_KEY: bool,
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
        {
            MULTI_MODELS_KEY: [
                {MODEL_ENTRY_PATH_KEY: And(str, len), MODEL_ENTRY_META_KEY: MODEL_SCHEMA.schema}
            ]
        }
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
        bool,
            Whether the given metadata is suspected to be a model metadata
        """

        return (
            isinstance(metadata, dict)
            and cls.MODEL_ID_KEY in metadata
            and not DeploymentSchema.is_single_deployment_schema(metadata)
        )

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
    def is_unstructured(cls, metadata):
        return metadata[ModelSchema.TARGET_TYPE_KEY] in [
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_REGRESSION_KEY,
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_BINARY_KEY,
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_MULTICLASS_KEY,
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_OTHER_KEY,
        ]

    @classmethod
    def validate_and_transform_single(cls, model_metadata):
        model_metadata = cls._validate_and_transform_single(cls.MODEL_SCHEMA, model_metadata)
        logger.debug(f"Model configuration is valid (id: {model_metadata[cls.MODEL_ID_KEY]}).")
        return model_metadata

    @classmethod
    def validate_and_transform_multi(cls, multi_models_metadata):
        multi_model_metadata = cls._validate_and_transform_multi(
            cls.MULTI_MODELS_SCHEMA, multi_models_metadata
        )
        logger.debug("Multi models configuration is valid.")
        return multi_model_metadata

    @classmethod
    def _next_single_transformed(cls, multi_transformed):
        for model_entry in multi_transformed[cls.MULTI_MODELS_KEY]:
            yield model_entry[cls.MODEL_ENTRY_META_KEY]

    @classmethod
    def _validate_mutual_exclusive_keys(cls, model_metadata):
        for binary_class_label_key in [cls.POSITIVE_CLASS_LABEL_KEY, cls.NEGATIVE_CLASS_LABEL_KEY]:
            mutual_exclusive_keys = {
                cls.PREDICTION_THRESHOLD_KEY,
                binary_class_label_key,
                cls.CLASS_LABELS_KEY,
            }
            if len(mutual_exclusive_keys & model_metadata.keys()) > 1:
                raise InvalidModelSchema(f"Only one of '{mutual_exclusive_keys}' keys is allowed.")

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
    def _validate_data_integrity(cls, model_metadata):
        if ModelSchema.TEST_KEY in model_metadata:
            skip_test_value = cls.get_value(
                model_metadata, ModelSchema.TEST_KEY, ModelSchema.TEST_SKIP_KEY
            )
            test_dataset_value = cls.get_value(
                model_metadata, ModelSchema.TEST_KEY, ModelSchema.TEST_DATA_KEY
            )
            if (
                not skip_test_value
                and not cls.is_unstructured(model_metadata)
                and not ObjectId.is_valid(test_dataset_value)
            ):
                raise InvalidModelSchema(
                    f"Test data is invalid. Please provide a valid catalog ID from DataRobot."
                )


class DeploymentSchema(SharedSchema):
    MULTI_DEPLOYMENTS_KEY = "datarobot_deployments"
    DEPLOYMENT_ID_KEY = "git_datarobot_deployment_id"

    PREDICTION_ENVIRONMENT_NAME_KEY = "prediction_environment_name"  # Optional
    ADDITIONAL_METADATA_KEY = "additional_metadata"  # Optional

    # Deployment settings are optional, POST + PATCH
    LABEL_KEY = "label"  # Settings, Optional
    DESCRIPTION_KEY = "description"  # Settings, Optional
    IMPORTANCE_KEY = "importance"  # Settings, Optional
    IMPORTANCE_CRITICAL_VALUE = "CRITICAL"
    IMPORTANCE_HIGH_VALUE = "HIGH"
    IMPORTANCE_MODERATE_VALUE = ("MODERATE",)
    IMPORTANCE_LOW_VALUE = ("LOW",)

    # PATCH Only
    ASSOCIATION_ID_KEY = "association_id"  # Settings, Optional
    ENABLE_TARGET_DRIFT_KEY = "enable_target_drift"  # Settings, Optional
    ENABLE_FEATURE_DRIFT_KEY = "enable_feature_drift"  # Settings, Optional
    ENABLE_PREDICTIONS_COLLECTION_KEY = "enable_predictions_collection"  # Settings, Optional
    ENABLE_ACTUALS = "enable_actuals"
    CHALLENGER_MODELS = "challenger_models"

    ENABLE_SEGMENT_ANALYSIS_KEY = "segment_analysis"  # Settings, Optional
    SEGMENT_ANALYSIS_ATTRIBUTES_KEY = (
        "segment_analysis_attributes"  # Settings.segment_analysis, Optional
    )

    # The 'DATASET_WITH_PARTITIONING_COLUMN_KEY' is mutually exclusive with the
    # 'TRAINING_DATASET_KEY' & 'HOLDOUT_DATASET_KEY'. The latter two are used with unstructured
    # models.
    DATASET_WITH_PARTITIONING_COLUMN_KEY = "dataset_with_partitioning_column"
    TRAINING_DATASET_KEY = "training_dataset"
    HOLDOUT_DATASET_KEY = "holdout_dataset"

    DEPLOYMENT_SCHEMA = Schema(
        {
            DEPLOYMENT_ID_KEY: And(str, len),
            SharedSchema.MODEL_ID_KEY: And(str, len),
            # fromCustomModel + fromModelPackage
            Optional(PREDICTION_ENVIRONMENT_NAME_KEY): And(str, len),
            Optional(SharedSchema.SETTINGS_SECTION_KEY): {
                Optional(LABEL_KEY): And(str, len),  # fromModelPackage
                Optional(ADDITIONAL_METADATA_KEY): {
                    And(str, len): And(str, len)
                },  # fromModelPackage
                Optional(DESCRIPTION_KEY): And(str, len),  # fromModelPackage
                Optional(IMPORTANCE_KEY): Or(  # fromModelPackage
                    IMPORTANCE_CRITICAL_VALUE,
                    IMPORTANCE_HIGH_VALUE,
                    IMPORTANCE_MODERATE_VALUE,
                    IMPORTANCE_LOW_VALUE,
                ),
                Optional(ASSOCIATION_ID_KEY): And(str, len),  # Update settings
                Optional(ENABLE_TARGET_DRIFT_KEY): bool,  # Update settings
                Optional(ENABLE_FEATURE_DRIFT_KEY): bool,  # Update settings
                Optional(ENABLE_PREDICTIONS_COLLECTION_KEY): bool,  # Update settings
                Optional(ENABLE_ACTUALS): bool,  # Update settings
                Optional(CHALLENGER_MODELS): bool,  # Update settings
                Optional(ENABLE_SEGMENT_ANALYSIS_KEY): bool,  # Update settings
                Optional(SEGMENT_ANALYSIS_ATTRIBUTES_KEY): And(  # Update settings
                    list, lambda l: all(len(a) > 0 for a in l)
                ),
                Optional(DATASET_WITH_PARTITIONING_COLUMN_KEY): And(
                    str, lambda i: ObjectId.is_valid(i)
                ),
                Optional(TRAINING_DATASET_KEY): And(str, lambda i: ObjectId.is_valid(i)),
                Optional(HOLDOUT_DATASET_KEY): And(str, lambda i: ObjectId.is_valid(i)),
            },
        }
    )
    MULTI_DEPLOYMENTS_SCHEMA = Schema([DEPLOYMENT_SCHEMA.schema])

    @classmethod
    def is_single_deployment_schema(cls, metadata):
        """
        Checks whether the given metadata is suspected to be a deployment metadata

        Parameters
        ----------
        metadata : dict
            A deployment metadata

        Returns
        -------
        bool,
            Whether the given metadata is suspected to be a single deployment metadata
        """

        return isinstance(metadata, dict) and cls.DEPLOYMENT_ID_KEY in metadata

    @classmethod
    def is_multi_deployments_schema(cls, metadata):
        """
        Checks whether the given metadata is a multi-deployments schema

        Parameters
        ----------
        metadata : list
            A multi-deployments metadata

        Returns
        -------
        bool,
            Whether the given metadata is suspected to be a a list of deployments metadata
        """

        return (
            isinstance(metadata, list)
            and metadata
            and isinstance(metadata[0], dict)
            and DeploymentSchema.DEPLOYMENT_ID_KEY in metadata[0]
        )

    @classmethod
    def validate_and_transform_single(cls, deployment_metadata):
        transformed = cls._validate_and_transform_single(cls.DEPLOYMENT_SCHEMA, deployment_metadata)
        logger.debug(
            f"Deployment configuration is valid (id: {transformed[cls.DEPLOYMENT_ID_KEY]})."
        )
        return transformed

    @classmethod
    def validate_and_transform_multi(cls, multi_deployments_metadata):
        transformed = cls._validate_and_transform_multi(
            cls.MULTI_DEPLOYMENTS_SCHEMA, multi_deployments_metadata
        )
        logger.debug("Multi deployments configuration is valid.")
        return transformed

    @classmethod
    def _next_single_transformed(cls, multi_transformed):
        for deployment_entry in multi_transformed:
            yield deployment_entry

    @classmethod
    def _validate_mutual_exclusive_keys(cls, deployment_metadata):
        settings_section = cls.get_value(deployment_metadata, DeploymentSchema.SETTINGS_SECTION_KEY)
        if not settings_section:
            return

        for dataset_for_unstructured in [cls.TRAINING_DATASET_KEY, cls.HOLDOUT_DATASET_KEY]:
            mutual_exclusive_keys = {
                dataset_for_unstructured,
                cls.DATASET_WITH_PARTITIONING_COLUMN_KEY,
            }
            if len(mutual_exclusive_keys & settings_section.keys()) > 1:
                raise InvalidModelSchema(f"Only one of '{mutual_exclusive_keys}' keys is allowed.")
