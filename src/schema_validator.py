#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A module that contains schema validators for model and deployment definitions.
"""

import logging

from bson import ObjectId
from schema import And
from schema import Optional
from schema import Or
from schema import Schema
from schema import SchemaError
from schema import Use

from common.convertors import MemoryConvertor
from common.exceptions import EmptyKey
from common.exceptions import InvalidModelSchema
from common.exceptions import InvalidSchema
from common.exceptions import UnexpectedType
from common.namepsace import Namespace

logger = logging.getLogger()


class SharedSchema:
    """
    A shared schema that contains attributes and methods that are shared between model and
    deployment schemas.
    """

    MODEL_ID_KEY = "user_provided_model_id"
    SETTINGS_SECTION_KEY = "settings"

    @classmethod
    def _validate_and_transform_single(cls, schema, metadata):
        try:
            transformed = schema.validate(metadata)
            cls._validate_single_transformed(transformed)
            return transformed
        except SchemaError as ex:
            raise InvalidSchema(ex.code) from ex

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
        except SchemaError as ex:
            raise InvalidSchema(ex.code) from ex

    @classmethod
    def _next_single_transformed(cls, multi_transformed):
        """
        A generator that must be implemented by a derived class. It returns the next
        metadata entry in a multi metadata definition.

        Parameters
        ----------
        multi_transformed : dict
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

    @staticmethod
    def get_value(metadata: dict, key, *sub_keys):
        """
        Extract a value from the metadata, for a given key hierarchy. The assumption is that parent
        keys are always dictionaries.

        Parameters
        ----------
        metadata : dict
            A model schema dictionary.
        key: str
           A top level metadata key.
        sub_keys : list[str]
            Optional. A variable number of strings, representing sub-keys under the 'key' argument.

        Returns
        -------
            A value or None
        """

        if not isinstance(metadata, dict):
            raise UnexpectedType(
                "Expecting first argument (metadata) to be a dict! "
                f"type: {type(metadata)}, value: '{metadata}'"
            )
        if not key:
            raise EmptyKey("An invalid empty key is provided to read a value from.")

        value = metadata.get(key)
        for sub_key in sub_keys:
            if not isinstance(value, dict):
                return None
            value = value.get(sub_key)
        return value

    @staticmethod
    def set_value(metadata: dict, key, *sub_keys, value):
        """
        Set a value in the metadata. If the key(s) do not exist, they'll be added.

        Parameters
        ----------
        metadata : dict
            The metadata.
        key: str
            A key name from the associated metadata schema.
        sub_keys: tuple
            Optional sub-keys, which are expected to reside under the top level key.
        value : Any
            A value to set

        Returns
        -------
        dict,
            The revised metadata after the value was set.
        """

        if not isinstance(metadata, dict):
            raise UnexpectedType(
                "Expecting first argument (metadata) to be a dict! "
                f"type: {type(metadata)}, value: '{metadata}'"
            )

        section = metadata
        keys = (key,) + sub_keys
        for a_key in keys[:-1]:
            if a_key not in section:
                section[a_key] = {}
            section = section[a_key]
            if not isinstance(section, dict):
                raise UnexpectedType(
                    f"A section in a metadata is expected to be a dict. Section: {section}"
                )
        section[keys[-1]] = value

        return metadata


class ModelSchema(SharedSchema):
    """
    A schema definition of a custom inference model.
    """

    MULTI_MODELS_KEY = "datarobot_models"
    MODEL_ENTRY_PATH_KEY = "model_path"
    MODEL_ENTRY_META_KEY = "model_metadata"

    TARGET_TYPE_KEY = "target_type"
    TARGET_TYPE_BINARY = "Binary"
    TARGET_TYPE_REGRESSION = "Regression"
    TARGET_TYPE_MULTICLASS = "Multiclass"
    TARGET_TYPE_ANOMALY_DETECTION = "Anomaly Detection"
    TARGET_TYPE_TEXT_GENERATION = "TextGeneration"
    TARGET_TYPE_UNSTRUCTURED_BINARY = "Unstructured (Binary)"
    TARGET_TYPE_UNSTRUCTURED_REGRESSION = "Unstructured (Regression)"
    TARGET_TYPE_UNSTRUCTURED_MULTICLASS = "Unstructured (Multiclass)"
    TARGET_TYPE_UNSTRUCTURED_OTHER = "Unstructured (Other)"

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

    # The 'PARTITIONING_COLUMN_KEY' is relevant for structured models only and is optional.
    PARTITIONING_COLUMN_KEY = "partitioning_column"
    TRAINING_DATASET_ID_KEY = "training_dataset_id"
    # The 'HOLDOUT_DATASET_ID_KEY' is relevant for unstructured models only and is optional.
    HOLDOUT_DATASET_ID_KEY = "holdout_dataset_id"

    VERSION_KEY = "version"
    MODEL_ENV_ID_KEY = "model_environment_id"
    INCLUDE_GLOB_KEY = "include_glob_pattern"
    EXCLUDE_GLOB_KEY = "exclude_glob_pattern"
    MEMORY_KEY = "memory"
    REPLICAS_KEY = "replicas"
    EGRESS_NETWORK_POLICY_KEY = "egress_network_policy"
    EGRESS_NETWORK_POLICY_NONE = "NONE"
    EGRESS_NETWORK_POLICY_PUBLIC = "PUBLIC"

    MODEL_REPLACEMENT_REASON_KEY = "model_replacement_reason"
    MODEL_REPLACEMENT_REASON_ACCURACY = "ACCURACY"
    MODEL_REPLACEMENT_REASON_DATA_DRIFT = "DATA_DRIFT"
    MODEL_REPLACEMENT_REASON_ERRORS = "ERRORS"
    MODEL_REPLACEMENT_REASON_SCHEDULED_REFRESH = "SCHEDULED_REFRESH"
    MODEL_REPLACEMENT_REASON_SCORING_SPEED = "SCORING_SPEED"
    MODEL_REPLACEMENT_REASON_DEPRECATION = "DEPRECATION"
    MODEL_REPLACEMENT_REASON_OTHER = "OTHER"

    TEST_KEY = "test"
    TEST_SKIP_KEY = "skip"
    TEST_DATA_ID_KEY = "test_data_id"

    CHECKS_KEY = "checks"
    NULL_VALUE_IMPUTATION_KEY = "null_value_imputation"
    CHECK_ENABLED_KEY = "enabled"
    BLOCK_DEPLOYMENT_IF_FAILS_KEY = "block_deployment_if_fails"
    SIDE_EFFECTS_KEY = "side_effects"
    PREDICTION_VERIFICATION_KEY = "prediction_verification"
    OUTPUT_DATASET_ID_KEY = "output_dataset_id"
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

    MODEL_REGISTRY_KEY = "model_registry"
    MODEL_NAME = "model_name"
    MODEL_DESCRIPTION = "model_description"
    GLOBAL = "global"

    MODEL_SCHEMA = Schema(
        {
            SharedSchema.MODEL_ID_KEY: And(str, len, Use(Namespace.namespaced)),
            TARGET_TYPE_KEY: Or(
                TARGET_TYPE_BINARY,
                TARGET_TYPE_REGRESSION,
                TARGET_TYPE_MULTICLASS,
                TARGET_TYPE_ANOMALY_DETECTION,
                TARGET_TYPE_TEXT_GENERATION,
                TARGET_TYPE_UNSTRUCTURED_BINARY,
                TARGET_TYPE_UNSTRUCTURED_REGRESSION,
                TARGET_TYPE_UNSTRUCTURED_MULTICLASS,
                TARGET_TYPE_UNSTRUCTURED_OTHER,
            ),
            SharedSchema.SETTINGS_SECTION_KEY: {
                NAME_KEY: And(str, len),
                Optional(DESCRIPTION_KEY): And(str, len),
                Optional(LANGUAGE_KEY): And(str, len),
                TARGET_NAME_KEY: And(str, len),
                Optional(PREDICTION_THRESHOLD_KEY): And(float, lambda n: 0 <= n <= 1),
                Optional(POSITIVE_CLASS_LABEL_KEY): And(str, len),
                Optional(NEGATIVE_CLASS_LABEL_KEY): And(str, len),
                Optional(CLASS_LABELS_KEY): list,
                Optional(PARTITIONING_COLUMN_KEY): And(str, len),
                Optional(TRAINING_DATASET_ID_KEY): And(str, ObjectId.is_valid),
                Optional(HOLDOUT_DATASET_ID_KEY): And(str, ObjectId.is_valid),
            },
            VERSION_KEY: {
                MODEL_ENV_ID_KEY: And(str, ObjectId.is_valid),
                Optional(INCLUDE_GLOB_KEY, default=[]): And(
                    list, lambda l: all(isinstance(e, str) and len(e) > 0 for e in l)
                ),
                Optional(EXCLUDE_GLOB_KEY, default=[]): And(
                    list, lambda l: all(isinstance(x, str) and len(x) > 0 for x in l)
                ),
                Optional(MEMORY_KEY): Use(MemoryConvertor.to_bytes),
                Optional(REPLICAS_KEY): And(int, lambda r: r > 0),
                Optional(EGRESS_NETWORK_POLICY_KEY): Or(
                    EGRESS_NETWORK_POLICY_NONE, EGRESS_NETWORK_POLICY_PUBLIC
                ),
                Optional(PARTITIONING_COLUMN_KEY): And(str, len),
                Optional(TRAINING_DATASET_ID_KEY): And(str, ObjectId.is_valid),
                Optional(HOLDOUT_DATASET_ID_KEY): And(str, ObjectId.is_valid),
                Optional(MODEL_REPLACEMENT_REASON_KEY, default=MODEL_REPLACEMENT_REASON_OTHER): Or(
                    MODEL_REPLACEMENT_REASON_ACCURACY,
                    MODEL_REPLACEMENT_REASON_DATA_DRIFT,
                    MODEL_REPLACEMENT_REASON_ERRORS,
                    MODEL_REPLACEMENT_REASON_SCHEDULED_REFRESH,
                    MODEL_REPLACEMENT_REASON_SCORING_SPEED,
                    MODEL_REPLACEMENT_REASON_DEPRECATION,
                    MODEL_REPLACEMENT_REASON_OTHER,
                ),
            },
            Optional(TEST_KEY): {
                # The skip attribute allows users to have the test section in their yaml file
                # and still disable testing
                Optional(TEST_SKIP_KEY, default=False): bool,
                Optional(TEST_DATA_ID_KEY): And(str, ObjectId.is_valid),
                Optional(MEMORY_KEY): Use(MemoryConvertor.to_bytes),
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
                        OUTPUT_DATASET_ID_KEY: And(str, ObjectId.is_valid),
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
            Optional(MODEL_REGISTRY_KEY): {
                Optional(MODEL_NAME): And(str, len),
                Optional(MODEL_DESCRIPTION): And(str, len),
                Optional(GLOBAL, default=False): bool,
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
        """
        Whether the given model's target type is binary.

        Parameters
        ----------
        metadata : dict
            A single model metadata.

        Returns
        -------
        bool,
            Whether the model's target type is Binary.
        """

        return metadata[ModelSchema.TARGET_TYPE_KEY] in [
            cls.TARGET_TYPE_BINARY,
            cls.TARGET_TYPE_UNSTRUCTURED_BINARY,
        ]

    @classmethod
    def is_regression(cls, metadata):
        """
        Whether the given model's target type is regression.

        Parameters
        ----------
        metadata : dict
            A single model metadata.

        Returns
        -------
        bool,
            Whether the model's target type is Regression.
        """

        return metadata[ModelSchema.TARGET_TYPE_KEY] in [
            cls.TARGET_TYPE_REGRESSION,
            cls.TARGET_TYPE_UNSTRUCTURED_REGRESSION,
        ]

    @classmethod
    def is_multiclass(cls, metadata):
        """
        Whether the given model's target type is multi-class.

        Parameters
        ----------
        metadata : dict
            A single model metadata.

        Returns
        -------
        bool,
            Whether the model's target type is MultiClass.
        """

        return metadata[ModelSchema.TARGET_TYPE_KEY] in [
            cls.TARGET_TYPE_MULTICLASS,
            cls.TARGET_TYPE_UNSTRUCTURED_MULTICLASS,
        ]

    @classmethod
    def is_unstructured(cls, metadata):
        """
        Whether the given model's target type is unstructured (vs. structured).

        Parameters
        ----------
        metadata : dict
            A single model metadata.

        Returns
        -------
        bool,
            Whether the model's target is unstructured.
        """

        return metadata[ModelSchema.TARGET_TYPE_KEY] in [
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_REGRESSION,
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_BINARY,
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_MULTICLASS,
            ModelSchema.TARGET_TYPE_UNSTRUCTURED_OTHER,
        ]

    @classmethod
    def validate_and_transform_single(cls, model_metadata):
        """
        Validate a single model metadata and run transformation to fill in derived and
        calculated attributes.

        Parameters
        ----------
        model_metadata : dict
            A single model metadata.

        Returns
        -------
        dict,
            A single model metadata.
        """

        model_metadata = cls._validate_and_transform_single(cls.MODEL_SCHEMA, model_metadata)
        logger.debug("Model configuration is valid (id: %s).", model_metadata[cls.MODEL_ID_KEY])
        return model_metadata

    @classmethod
    def validate_and_transform_multi(cls, multi_models_metadata):
        """
        Validate and multi-model metadata and run transformation to fill in derived and
        calculated attributes.

        Parameters
        ----------
        multi_models_metadata : dict
            A multi-models metadata definition.

        Returns
        -------
        dict,
            A mutli-model metadata.
        """

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
    def _validate_mutual_exclusive_keys(cls, single_transformed_metadata):
        settings_section = single_transformed_metadata[ModelSchema.SETTINGS_SECTION_KEY]
        for binary_class_label_key in [cls.POSITIVE_CLASS_LABEL_KEY, cls.NEGATIVE_CLASS_LABEL_KEY]:
            mutual_exclusive_keys = {
                cls.PREDICTION_THRESHOLD_KEY,
                binary_class_label_key,
                cls.CLASS_LABELS_KEY,
            }
            if len(mutual_exclusive_keys & settings_section.keys()) > 1:
                raise InvalidModelSchema(f"Only one of '{mutual_exclusive_keys}' keys is allowed.")

        # Check training/holdout mutually exclusive at model level
        mutual_exclusive_keys = {cls.PARTITIONING_COLUMN_KEY, cls.HOLDOUT_DATASET_ID_KEY}
        if len(mutual_exclusive_keys & settings_section.keys()) > 1:
            raise InvalidModelSchema(
                f"Only one of '{mutual_exclusive_keys}' keys is allowed in settings section."
            )

        # Check training/holdout mutually exclusive at model version level
        version_section = single_transformed_metadata[ModelSchema.VERSION_KEY]
        if len(mutual_exclusive_keys & version_section.keys()) > 1:
            raise InvalidModelSchema(
                f"Only one of '{mutual_exclusive_keys}' keys is allowed in version section."
            )

        # Check training/holdout mutually exclusive between model and version levels
        mutual_exclusive_keys = {
            cls.PARTITIONING_COLUMN_KEY,
            cls.TRAINING_DATASET_ID_KEY,
            cls.HOLDOUT_DATASET_ID_KEY,
        }
        if (
            len(mutual_exclusive_keys & settings_section.keys()) > 0
            and len(mutual_exclusive_keys & version_section.keys()) > 0
        ):
            raise InvalidModelSchema(
                f"Definition of '{mutual_exclusive_keys}' keys are either allowed under settings "
                "or version sections."
            )

    @classmethod
    def _validate_dependent_keys(cls, single_transformed_metadata):
        settings_section = single_transformed_metadata[ModelSchema.SETTINGS_SECTION_KEY]
        if cls.is_binary(single_transformed_metadata):
            binary_label_keys = {
                ModelSchema.POSITIVE_CLASS_LABEL_KEY,
                ModelSchema.NEGATIVE_CLASS_LABEL_KEY,
            }
            if len(binary_label_keys & set(settings_section.keys())) != 2:
                raise InvalidModelSchema(
                    f"Binary model must be defined with the '{binary_label_keys}' keys."
                )
        elif (
            cls.is_multiclass(single_transformed_metadata)
            and settings_section.get(ModelSchema.CLASS_LABELS_KEY) is None
        ):
            raise InvalidModelSchema(
                "Multiclass model must be define with the 'mapping_classes' key."
            )

        stability = (
            single_transformed_metadata.get(ModelSchema.TEST_KEY, {})
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
    def _validate_data_integrity(cls, single_transformed_metadata):
        if ModelSchema.TEST_KEY in single_transformed_metadata:
            skip_test_value = cls.get_value(
                single_transformed_metadata, ModelSchema.TEST_KEY, ModelSchema.TEST_SKIP_KEY
            )
            test_dataset_value = cls.get_value(
                single_transformed_metadata, ModelSchema.TEST_KEY, ModelSchema.TEST_DATA_ID_KEY
            )
            if (
                not skip_test_value
                and not cls.is_unstructured(single_transformed_metadata)
                and not ObjectId.is_valid(test_dataset_value)
            ):
                raise InvalidModelSchema(
                    "Test data is invalid. Please provide a valid catalog ID from DataRobot."
                )


class DeploymentSchema(SharedSchema):
    """
    A schema definition of a custom inference model deployment.
    """

    MULTI_DEPLOYMENTS_KEY = "datarobot_deployments"
    DEPLOYMENT_ID_KEY = "user_provided_deployment_id"

    PREDICTION_ENVIRONMENT_NAME_KEY = "prediction_environment_name"  # Optional
    ADDITIONAL_METADATA_KEY = "additional_metadata"  # Optional

    # Deployment settings are optional, POST + PATCH
    LABEL_KEY = "label"  # Settings, Optional
    DESCRIPTION_KEY = "description"  # Settings, Optional
    IMPORTANCE_KEY = "importance"  # Settings, Optional
    IMPORTANCE_CRITICAL = "CRITICAL"
    IMPORTANCE_HIGH = "HIGH"
    IMPORTANCE_MODERATE = "MODERATE"
    IMPORTANCE_LOW = "LOW"

    ENABLE_TARGET_DRIFT_KEY = "enable_target_drift"  # Settings, Optional
    ENABLE_FEATURE_DRIFT_KEY = "enable_feature_drift"  # Settings, Optional
    ENABLE_PREDICTIONS_COLLECTION_KEY = "enable_predictions_collection"  # Settings, Optional

    ASSOCIATION_KEY = "association"
    ASSOCIATION_ASSOCIATION_ID_COLUMN_KEY = "association_id_column"
    ASSOCIATION_REQUIRED_IN_PRED_REQUEST_KEY = "required_in_pred_request"
    ASSOCIATION_ACTUAL_VALUES_COLUMN_KEY = "actual_values_column"
    ASSOCIATION_ACTUALS_DATASET_ID_KEY = "actuals_dataset_id"

    ENABLE_CHALLENGER_MODELS_KEY = "enable_challenger_models"

    SEGMENT_ANALYSIS_KEY = "segment_analysis"  # Settings, Optional
    ENABLE_SEGMENT_ANALYSIS_KEY = "enabled"
    SEGMENT_ANALYSIS_ATTRIBUTES_KEY = "attributes"  # Settings.segment_analysis, Optional

    DEPLOYMENT_SCHEMA = Schema(
        {
            DEPLOYMENT_ID_KEY: And(str, len, Use(Namespace.namespaced)),
            SharedSchema.MODEL_ID_KEY: And(str, len, Use(Namespace.namespaced)),
            # fromCustomModel + fromModelPackage
            Optional(PREDICTION_ENVIRONMENT_NAME_KEY): And(str, len),
            Optional(SharedSchema.SETTINGS_SECTION_KEY): {
                Optional(LABEL_KEY): And(str, len),  # fromModelPackage
                Optional(ADDITIONAL_METADATA_KEY): {
                    And(str, len): And(str, len)
                },  # fromModelPackage
                Optional(DESCRIPTION_KEY): And(str, len),  # fromModelPackage
                # NOTE: a higher importance value than "LOW" will trigger a review process for any
                # operation, such as 'create', 'update', 'delete', etc. So, practically
                # the user will need to wait for approval from a reviewer in order to be able
                # to apply new changes and merge them to the main branch.
                Optional(IMPORTANCE_KEY, default=IMPORTANCE_LOW): Or(  # fromModelPackage
                    IMPORTANCE_CRITICAL, IMPORTANCE_HIGH, IMPORTANCE_MODERATE, IMPORTANCE_LOW
                ),
                Optional(ASSOCIATION_KEY): {
                    Optional(ASSOCIATION_ASSOCIATION_ID_COLUMN_KEY): And(str, len),
                    Optional(ASSOCIATION_REQUIRED_IN_PRED_REQUEST_KEY): bool,
                    # NOTE: The ACTUALS dataset is submitted when the deployment is created.
                    # If the user changes it afterwards in his yaml definition, it'll not be
                    # uploaded, unless the `ASSOCIATION_ASSOCIATION_ID_COLUMN_KEY` will be changed
                    # too.
                    Optional(ASSOCIATION_ACTUAL_VALUES_COLUMN_KEY): And(str, len),
                    Optional(ASSOCIATION_ACTUALS_DATASET_ID_KEY): And(str, ObjectId.is_valid),
                },
                Optional(ENABLE_TARGET_DRIFT_KEY): bool,  # Update settings
                Optional(ENABLE_FEATURE_DRIFT_KEY): bool,  # Update settings
                Optional(ENABLE_PREDICTIONS_COLLECTION_KEY): bool,  # Update settings
                Optional(ENABLE_CHALLENGER_MODELS_KEY): bool,  # Update settings
                Optional(SEGMENT_ANALYSIS_KEY): {
                    ENABLE_SEGMENT_ANALYSIS_KEY: bool,
                    Optional(SEGMENT_ANALYSIS_ATTRIBUTES_KEY): And(
                        list, lambda l: all(len(a) > 0 for a in l)
                    ),
                },
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
        """
        Validate a single deployment metadata and run transformation to fill in derived and
        calculated attributes.

        Parameters
        ----------
        deployment_metadata : dict
            A single deployment metadata.

        Returns
        -------
        dict,
            A single deployment metadata.

        """
        transformed = cls._validate_and_transform_single(cls.DEPLOYMENT_SCHEMA, deployment_metadata)
        logger.debug(
            "Deployment configuration is valid (id: %s).", transformed[cls.DEPLOYMENT_ID_KEY]
        )
        return transformed

    @classmethod
    def validate_and_transform_multi(cls, multi_deployments_metadata):
        """
        Validate a multi-deployments metadata and run transformation to fill in derived and
        calculated attributes.

        Parameters
        ----------
        multi_deployments_metadata : dict
            A multi-deployments metadata.

        Returns
        -------
        dict,
            A multi-deployments metadata.

        """
        transformed = cls._validate_and_transform_multi(
            cls.MULTI_DEPLOYMENTS_SCHEMA, multi_deployments_metadata
        )
        logger.debug("Multi deployments configuration is valid.")
        return transformed

    @classmethod
    def _next_single_transformed(cls, multi_transformed):
        for deployment_entry in multi_transformed:
            yield deployment_entry
