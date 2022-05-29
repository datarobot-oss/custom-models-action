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
    DEPLOYMENT_ID_KEY = "git_datarobot_deployment_id"
    VERSION_KEY = "version"
    INCLUDE_GLOB_KEY = "include_glob_pattern"
    EXCLUDE_GLOB_KEY = "exclude_glob_pattern"

    MODEL_SCHEMA = Schema(
        {
            MODEL_ID_KEY: str,
            Optional(DEPLOYMENT_ID_KEY): str,
            "target_type": Or(
                "Binary",
                "Regression",
                "Multiclass",
                "Anomaly Detection",
                "Unstructured (Binary)",
                "Unstructured (Regression)",
                "Unstructured (Multiclass)",
                "Unstructured (Other)",
            ),
            "target_name": str,
            Optional("prediction_threshold"): And(float, lambda n: 0 <= n <= 1),
            Optional("positive_class_label"): str,
            Optional("negative_class_label"): str,
            Optional("mapping_classes"): list,
            Optional("language"): str,
            Optional("settings"): {
                Optional("name"): str,
                Optional("description"): str,
                Optional("training_dataset"): And(str, lambda i: ObjectId.is_valid(i)),
                Optional("holdout_dataset"): And(str, lambda i: ObjectId.is_valid(i)),
            },
            VERSION_KEY: {
                "model_environment": And(str, lambda i: ObjectId.is_valid(i)),
                Optional(INCLUDE_GLOB_KEY, default=[]): And(
                    list, lambda l: all(isinstance(e, str) for e in l)
                ),
                Optional(EXCLUDE_GLOB_KEY, default=[]): And(
                    list, lambda l: all(isinstance(x, str) for x in l)
                ),
                Optional("memory"): Use(lambda v: MemoryConvertor.to_bytes(v)),
                Optional("replicas"): And(int, lambda r: r > 0),
            },
            Optional("test"): {
                "test_data": And(str, lambda i: ObjectId.is_valid(i)),
                Optional("memory"): Use(lambda v: MemoryConvertor.to_bytes(v)),
                Optional("checks"): {
                    Optional("null_imputation"): {
                        "value": Or("yes", "no"),
                        "block_deployment_if_fails": Or("yes", "no"),
                    },
                    Optional("side_effect"): {
                        "value": Or("yes", "no"),
                        "block_deployment_if_fails": Or("yes", "no"),
                    },
                    Optional("prediction_verification"): {
                        "value": Or("yes", "no"),
                        "block_deployment_if_fails": Or("yes", "no"),
                    },
                    Optional("prediction_verification"): {
                        "value": Or("yes", "no"),
                        "block_deployment_if_fails": Or("yes", "no"),
                        "output_dataset": And(str, lambda i: ObjectId.is_valid(i)),
                        Optional("match_threshold"): And(float, lambda v: 0 <= v <= 1),
                        Optional("passing_match_rate"): And(int, lambda v: 0 <= v <= 100),
                    },
                    Optional("performance"): {
                        "value": Or("yes", "no"),
                        "block_deployment_if_fails": Or("yes", "no"),
                        Optional("maximum_response_time"): And(int, lambda v: 1 <= v <= 1800),
                        Optional("check_duration_limit"): And(int, lambda v: 1 <= v <= 1800),
                        Optional("number_of_parallel_users"): And(int, lambda v: 1 <= v <= 4),
                    },
                    Optional("stability"): {
                        "value": Or("yes", "no"),
                        "block_deployment_if_fails": Or("yes", "no"),
                        Optional("total_prediction_requests"): And(int, lambda v: v >= 1),
                        Optional("passing_rate"): And(int, lambda v: 0 <= v <= 100),
                        Optional("number_of_parallel_users"): And(int, lambda v: 1 <= v <= 4),
                        Optional("minimum_payload_size"): And(int, lambda v: v >= 1),
                        Optional("maximum_payload_size"): And(int, lambda v: v >= 1),
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

    def validate_and_transform_single(self, model_metadata):
        try:
            transformed = self.MODEL_SCHEMA.validate(model_metadata)
            self._validate_single_model(transformed)
            return transformed
        except SchemaError as se:
            raise InvalidModelSchema(se.code)

    def _validate_single_model(self, single_model_metadata):
        self._validate_mutual_exclusive_keys(single_model_metadata)
        self._validate_dependent_keys(single_model_metadata)
        logger.debug(
            f"Model configuration is valid (id: {single_model_metadata[self.MODEL_ID_KEY]})."
        )

    def validate_and_transform_multi(self, multi_models_metadata):
        # Validates and transform
        try:
            transformed = self.MULTI_MODELS_SCHEMA.validate(multi_models_metadata)
            for model_entry in transformed[self.MULTI_MODELS_KEY]:
                self._validate_single_model(model_entry[self.MODEL_ENTRY_META_KEY])
            return transformed
        except SchemaError as se:
            raise InvalidModelSchema(se.code)

    @staticmethod
    def _validate_mutual_exclusive_keys(model_metadata):
        for binary_class_label_key in ["positive_class_label", "negative_class_label"]:
            mutual_exclusive_keys = {
                "prediction_threshold",
                binary_class_label_key,
                "mapping_classes",
            }
            if len(mutual_exclusive_keys & model_metadata.keys()) > 1:
                raise InvalidModelSchema(f"Only one of '{mutual_exclusive_keys}' keys is expected")

    @staticmethod
    def _validate_dependent_keys(model_metadata):
        model_target_type = model_metadata["target_type"]
        if model_target_type == "Binary":
            binary_label_keys = {"positive_class_label", "negative_class_label"}
            if len(binary_label_keys & set(model_metadata.keys())) != 2:
                raise InvalidModelSchema(
                    f"Binary model must be defined with the '{binary_label_keys}' keys."
                )
        elif model_target_type == "Multiclass" and model_metadata.get("mapping_classes") is None:
            raise InvalidModelSchema(
                f"Multiclass model must be define with the 'mapping_classes' key."
            )

        stability = model_metadata.get("test", {}).get("checks", {}).get("stability", {})
        if stability:
            minimum_payload_size = stability.get("minimum_payload_size", 1)
            maximum_payload_size = stability.get("maximum_payload_size", 1000)
            if maximum_payload_size < minimum_payload_size:
                raise InvalidModelSchema(
                    f"Stability test check minimum payload size ({minimum_payload_size}) "
                    f"is higher than the maximum ({maximum_payload_size})"
                )
