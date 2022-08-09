#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A lightweight client that provides an interface to interact with DataRobot application. It
communicates with DataRobot via the published public API.
"""

import json
import logging
import time
from collections import namedtuple

from requests_toolbelt import MultipartEncoder

from common import constants
from common.exceptions import DataRobotClientError
from common.exceptions import HttpRequesterException
from common.exceptions import IllegalModelDeletion
from common.http_requester import HttpRequester
from common.string_util import StringUtil
from dr_api_attrs import DrApiAttrs
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema

logger = logging.getLogger(__name__)


class DrClient:
    """An implementation of the lightweight client"""

    CUSTOM_MODELS_ROUTE = "customModels/"
    CUSTOM_MODEL_ROUTE = CUSTOM_MODELS_ROUTE + "{model_id}/"
    CUSTOM_MODELS_VERSIONS_ROUTE = CUSTOM_MODEL_ROUTE + "versions/"
    CUSTOM_MODELS_VERSION_ROUTE = CUSTOM_MODEL_ROUTE + "versions/{model_ver_id}/"
    CUSTOM_MODELS_TEST_ROUTE = "customModelTests/"
    CUSTOM_MODEL_DEPLOYMENTS_ROUTE = "customModelDeployments/"
    CUSTOM_MODEL_TRAINING_DATA = CUSTOM_MODEL_ROUTE + "trainingData/"
    DATASETS_ROUTE = "datasets/"
    DATASET_UPLOAD_ROUTE = DATASETS_ROUTE + "fromFile/"
    MODEL_PACKAGES_CREATE_ROUTE = "modelPackages/fromCustomModelVersion/"
    DEPLOYMENTS_ROUTE = "deployments/"
    DEPLOYMENTS_CREATE_ROUTE = "deployments/fromModelPackage/"
    DEPLOYMENT_ROUTE = "deployments/{deployment_id}/"
    DEPLOYMENT_SETTINGS_ROUTE = DEPLOYMENT_ROUTE + "settings/"
    PREDICTION_ENVIRONMENTS_ROUTE = "predictionEnvironments/?supportedModelFormats=customModel"
    DEPLOYMENT_MODEL_ROUTE = DEPLOYMENT_ROUTE + "model/"
    DEPLOYMENT_MODEL_VALIDATION_ROUTE = DEPLOYMENT_MODEL_ROUTE + "validation/"
    DEPLOYMENT_MODEL_CHALLENGER_ROUTE = DEPLOYMENT_ROUTE + "challengers/"
    DEPLOYMENT_ACTUALS_UPDATE_ROUTE = DEPLOYMENT_ROUTE + "actuals/fromDataset/"

    MODEL_TARGET_TYPE_MAP = {
        ModelSchema.TARGET_TYPE_BINARY_KEY: "Binary",
        ModelSchema.TARGET_TYPE_UNSTRUCTURED_BINARY_KEY: "Binary",
        ModelSchema.TARGET_TYPE_REGRESSION_KEY: "Regression",
        ModelSchema.TARGET_TYPE_UNSTRUCTURED_REGRESSION_KEY: "Regression",
        ModelSchema.TARGET_TYPE_MULTICLASS_KEY: "Multiclass",
        ModelSchema.TARGET_TYPE_UNSTRUCTURED_MULTICLASS_KEY: "Multiclass",
        ModelSchema.TARGET_TYPE_UNSTRUCTURED_OTHER_KEY: "Unstructured",
    }

    MODEL_SETTINGS_KEYS_MAP = {
        ModelSchema.NAME_KEY: "name",
        ModelSchema.DESCRIPTION_KEY: "description",
        ModelSchema.LANGUAGE_KEY: "language",
        ModelSchema.TARGET_NAME_KEY: "targetName",
        ModelSchema.PREDICTION_THRESHOLD_KEY: "predictionThreshold",
        ModelSchema.POSITIVE_CLASS_LABEL_KEY: "positiveClassLabel",
        ModelSchema.NEGATIVE_CLASS_LABEL_KEY: "negativeClassLabel",
        ModelSchema.CLASS_LABELS_KEY: "classLabels",
    }

    def __init__(self, datarobot_webserver, datarobot_api_token, verify_cert=True):
        if "v2" not in datarobot_webserver:
            datarobot_webserver = f"{StringUtil.slash_suffix(datarobot_webserver)}api/v2/"

        self._http_requester = HttpRequester(datarobot_webserver, datarobot_api_token, verify_cert)

    def _wait_for_async_resolution(self, async_location, max_wait=600, return_on_completed=False):
        start_time = time.time()

        while time.time() < start_time + max_wait:
            response = self._http_requester.get(async_location, raw=True, allow_redirects=False)
            if response.status_code == 303:
                return response.headers["Location"]

            data = response.json()
            if (
                return_on_completed
                and response.status_code == 200
                and data["status"].lower() == "completed"
            ):
                return data

            if response.status_code != 200:
                raise HttpRequesterException(
                    f"Failed waiting for async resolution. "
                    f"Response code: {response.status_code}. "
                    f"Response body: {data}."
                )

            if data["status"].lower()[:5] in ["abort", "error"]:
                raise HttpRequesterException(f"Async task failed: {data}.")

            time.sleep(1)

        raise HttpRequesterException(f"Client timed out waiting for {async_location} to resolve.")

    def is_accessible(self):
        """Checks whether DataRobot application is accessible over the network"""

        logger.debug("Check if webserver is accessible ...")
        response = self._http_requester.get(
            f"{self._http_requester.webserver_api_path}/ping", raw=True
        )
        return response.status_code == 200 and response.json()["response"] == "pong"

    def fetch_custom_models(self):
        """
        Retrieve custom models from DataRobot.

        Returns
        -------
        list[dict],
            A list of DataRobot custom models.
        """

        logger.debug("Fetching custom models...")
        return self._paginated_fetch(self.CUSTOM_MODELS_ROUTE)

    def fetch_custom_model_by_git_id(self, git_model_id):
        """
        Retrieve a single custom model from DataRobot, given a Git model ID.

        Parameters
        ----------
        git_model_id : str
            A unique ID that is defined by the user.

        Returns
        -------
        dict or None,
            A DataRobot custom model dictionary or None if not found.
        """

        custom_models = self.fetch_custom_models()
        try:
            return next(cm for cm in custom_models if cm.get("gitModelId") == git_model_id)
        except StopIteration:
            return None

    def _paginated_fetch(self, route_url, **kwargs):
        def _fetch_single_page(url, raw):
            response = self._http_requester.get(url, raw, **kwargs)
            if response.status_code != 200:
                raise DataRobotClientError(
                    f"Failed to fetch entities of a single page. "
                    f"Response status: {response.status_code} "
                    f"Response body: {response.text}",
                    code=response.status_code,
                )

            response_json = response.json()
            _total_count = response_json["totalCount"]
            _page_count = response_json["count"]
            _next_page = response_json["next"]
            _returned_models = response_json["data"]
            logger.debug(
                f"Total: {_total_count}, page count: {_page_count}, Next page: {_next_page}"
            )
            return _returned_models, _next_page

        returned_entities, next_page = _fetch_single_page(route_url, raw=False)
        total_entities = returned_entities
        while next_page:
            returned_entities, next_page = _fetch_single_page(next_page, raw=True)
            total_entities.extend(returned_entities)

        return total_entities

    def create_custom_model(self, model_info):
        """
        Create a custom model in DataRobot.

        Parameters
        ----------
        model_info : ModelInfo
            A local model info as loaded from the local source tree.

        Returns
        -------
        dict,
            A DataRobot custom model entity.
        """

        payload = self._setup_payload_for_custom_model_creation(model_info)
        response = self._http_requester.post(self.CUSTOM_MODELS_ROUTE, json=payload)
        if response.status_code != 201:
            raise DataRobotClientError(
                f"Failed to create custom model. "
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
                code=response.status_code,
            )

        custom_model = response.json()
        logger.debug(f"Custom model created successfully (id: {custom_model['id']})")
        return custom_model

    @classmethod
    def _setup_payload_for_custom_model_creation(cls, model_info):
        target_type = model_info.get_value(ModelSchema.TARGET_TYPE_KEY)

        payload = {
            "customModelType": constants.CUSTOM_MODEL_TYPE,
            "targetType": cls.MODEL_TARGET_TYPE_MAP.get(target_type),
            "targetName": model_info.get_settings_value(ModelSchema.TARGET_NAME_KEY),
            "isUnstructuredModelKind": model_info.is_unstructured,
            "gitModelId": model_info.get_value(ModelSchema.MODEL_ID_KEY),
        }

        name = model_info.get_settings_value(ModelSchema.NAME_KEY)
        if name:
            payload["name"] = name

        description = model_info.get_settings_value(ModelSchema.DESCRIPTION_KEY)
        if description:
            payload["description"] = description

        lang = model_info.get_settings_value(ModelSchema.LANGUAGE_KEY)
        if lang:
            payload["language"] = lang

        if model_info.is_regression:
            regression_threshold = model_info.get_settings_value(
                ModelSchema.PREDICTION_THRESHOLD_KEY
            )
            if regression_threshold is not None:
                payload["predictionThreshold"] = regression_threshold
        elif model_info.is_binary:
            payload.update(
                {
                    "positiveClassLabel": model_info.get_settings_value(
                        ModelSchema.POSITIVE_CLASS_LABEL_KEY
                    ),
                    "negativeClassLabel": model_info.get_settings_value(
                        ModelSchema.NEGATIVE_CLASS_LABEL_KEY
                    ),
                }
            )
        elif model_info.is_multiclass:
            payload["classLabels"] = model_info.get_settings_value(ModelSchema.CLASS_LABELS_KEY)

        return payload

    def fetch_custom_model_versions(self, custom_model_id, **kwargs):
        """
        Retrieve DataRobot custom model versions for a given custom model ID.

        Parameters
        ----------
        custom_model_id : str
            A DataRobot custom model ID.
        kwargs : dict
            A key-value pairs to be submitted as additional attributes when querying DataRobot.

        Returns
        -------
        list[dict],
            A list of DataRobot custom model versions.
        """

        logger.debug(f"Fetching custom model versions for model '{custom_model_id}' ...")
        return self._paginated_fetch(
            self.CUSTOM_MODELS_VERSIONS_ROUTE.format(model_id=custom_model_id), **kwargs
        )

    def fetch_custom_model_version(self, custom_model_id, custom_model_version_id):
        """
        Retrieve a specific custom model version, given model ID and version ID.

        Parameters
        ----------
        custom_model_id : str
            A custom model ID
        custom_model_version_id :
            A custom model version ID

        Returns
        -------
        dict,
            A DataRobot custom model version
        """

        logger.debug(
            f"Fetching custom model version '{custom_model_version_id}' "
            f"for model '{custom_model_id}' ..."
        )
        url = self.CUSTOM_MODELS_VERSION_ROUTE.format(
            model_id=custom_model_id, model_ver_id=custom_model_version_id
        )
        response = self._http_requester.get(url)
        if response.status_code != 200:
            raise DataRobotClientError(
                f"Failed to get custom model version {custom_model_version_id} "
                f"of model {custom_model_id}. "
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
                code=response.status_code,
            )
        return response.json()

    def fetch_custom_model_latest_version_by_git_model_id(self, git_model_id):
        """
        Retrieve the latest custom model version, given a Git model ID.

        Parameters
        ----------
        git_model_id : str
            A unique ID that is defined by the user.

        Returns
        -------
        dict or None,
            A DataRobot custom model version if found, otherwise None.
        """

        logger.debug(f"Fetching custom model versions for git model '{git_model_id}' ...")

        custom_model = self.fetch_custom_model_by_git_id(git_model_id)
        if not custom_model:
            return None

        cm_versions = self.fetch_custom_model_versions(custom_model["id"], json={"limit": 1})
        if not cm_versions:
            return None

        return cm_versions[0]

    def create_custom_model_version(
        self,
        custom_model_id,
        model_info,
        ref_name,
        commit_url,
        main_branch_commit_sha,
        pull_request_commit_sha=None,
        changed_file_paths=None,
        file_ids_to_delete=None,
        from_latest=False,
    ):
        """
        Create a custom model version in DataRobot.

        Parameters
        ----------
        custom_model_id : str
            A DataRobot custom model ID.
        model_info : ModelInfo
            An information about the model in the local source tree.
        ref_name : str
            The branch or tag name that triggered the workflow run.
        commit_url : str
            A GitHub commit URL.
        main_branch_commit_sha : str
            A commit SHA from the main branch. For pull requests, it is the split point.
        pull_request_commit_sha : str or None
            The top commit sha of a feature branch in a pull request. Otherwise, None.
        changed_file_paths : list[ModelFilePath] or None
            A list of changed files related to the last GitHub action.
        file_ids_to_delete : list[str] or None
            A list of file IDs of DataRobot items to be deleted.
        from_latest : bool
            Whether to create the new version by first copying all the stuf from a previous
            version, or to create it from scratch.

        Returns
        -------
        str,
            A DataRobot version ID.
        """

        file_objs = []
        try:
            base_env_id = model_info.get_value(ModelSchema.VERSION_KEY, ModelSchema.MODEL_ENV_KEY)
            payload, file_objs = self._setup_payload_for_custom_model_version_creation(
                model_info,
                ref_name,
                commit_url,
                main_branch_commit_sha,
                pull_request_commit_sha,
                changed_file_paths,
                file_ids_to_delete=file_ids_to_delete,
                base_env_id=base_env_id,
            )
            mp = MultipartEncoder(fields=payload)
            headers = {"Content-Type": mp.content_type}

            url = self.CUSTOM_MODELS_VERSIONS_ROUTE.format(model_id=custom_model_id)
            if from_latest:
                response = self._http_requester.patch(url, data=mp, headers=headers)
            else:
                response = self._http_requester.post(url, data=mp, headers=headers)
        finally:
            for file_obj in file_objs:
                file_obj.close()

        if response.status_code != 201:
            raise DataRobotClientError(
                "Failed to create custom model version "
                f"({'from latest' if from_latest else 'new'}). "
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
                code=response.status_code,
            )

        version_id = response.json()["id"]
        logger.info(f"Custom model version created successfully (id: {version_id})")
        return version_id

    @classmethod
    def _setup_payload_for_custom_model_version_creation(
        cls,
        model_info,
        ref_name,
        commit_url,
        main_branch_commit_sha,
        pull_request_commit_sha,
        changed_file_paths,
        file_ids_to_delete=None,
        base_env_id=None,
    ):
        payload = [
            (
                "gitModelVersion",
                json.dumps(
                    {
                        "refName": ref_name,
                        "commitUrl": commit_url,
                        "mainBranchCommitSha": main_branch_commit_sha,
                        "pullRequestCommitSha": pull_request_commit_sha,
                    }
                ),
            ),
        ]

        file_objs = cls._setup_model_version_files(changed_file_paths, file_ids_to_delete, payload)

        is_major = True if changed_file_paths or file_ids_to_delete or base_env_id else False
        payload.append(("isMajorUpdate", str(is_major)))

        if base_env_id:
            payload.append(("baseEnvironmentId", base_env_id))

        memory = model_info.get_value(ModelSchema.VERSION_KEY, ModelSchema.MEMORY_KEY)
        if memory:
            payload.append(("maximumMemory", str(memory)))

        replicas = model_info.get_value(ModelSchema.VERSION_KEY, ModelSchema.REPLICAS_KEY)
        if replicas:
            payload.append(("replicas", str(replicas)))

        return payload, file_objs

    @staticmethod
    def _setup_model_version_files(changed_file_paths, file_ids_to_delete, payload):
        file_objs = []
        for model_filepath in changed_file_paths or []:
            fd = open(model_filepath.resolved, "rb")
            file_objs.append(fd)
            path_under_model = str(model_filepath.under_model)

            payload.append(("file", (path_under_model, fd)))
            payload.append(("filePath", path_under_model))

        for file_id_to_delete in file_ids_to_delete or []:
            payload.append(("filesToDelete", file_id_to_delete))

        return file_objs

    def delete_custom_model_by_model_id(self, custom_model_id):
        """
        Delete a custom model in DataRobot, given a DataRobot model ID.

        Parameters
        ----------
        custom_model_id : str
            A DataRobot custom inference model ID.
        """

        sub_path = f"{self.CUSTOM_MODELS_ROUTE}{custom_model_id}/"
        response = self._http_requester.delete(sub_path)
        if response.status_code != 204:
            raise DataRobotClientError(
                f"Failed to delete custom model. Error: {response.text}.",
                code=response.status_code,
            )

    def delete_custom_model_by_git_model_id(self, git_model_id):
        """
        Delete a custom model in DataRobot, given a Git model ID.

        Parameters
        ----------
        git_model_id : str
            A unique ID that is defined by the user.
        """

        test_custom_model = self.fetch_custom_model_by_git_id(git_model_id)
        if not test_custom_model:
            raise IllegalModelDeletion(
                f"Given custom model does not exist. git_model_id: {git_model_id}."
            )
        self.delete_custom_model_by_model_id(test_custom_model["id"])

    def run_custom_model_version_testing(self, model_id, model_version_id, model_info):
        """
        Post a query to start custom model version testing.

        Parameters
        ----------
        model_id : str
            A DataRobot custom model ID.
        model_version_id : str
            A DataRobot custom model version ID.
        model_info : ModelInfo
            A class that contains full information about a single model from the local source tree.
        """

        response = self._post_custom_model_test_request(model_id, model_version_id, model_info)
        location = self._wait_for_async_resolution(response.headers["Location"])
        response = self._http_requester.get(location, raw=True)
        response_data = response.json()

        if response_data["overallStatus"] != "succeeded":
            for check, result in response_data["testingStatus"]:
                status = result["status"]
                if status != "succeeded":
                    raise DataRobotClientError(
                        f"Custom model version check failed.\nCheck: '{check}'.\nStatus: {status}."
                        f"\nMessage: {result['message']}"
                    )
        logger.debug(
            f"Custom model testing pass with success. Git model id: {model_info.git_model_id}"
        )

    def _post_custom_model_test_request(self, model_id, model_version_id, model_info):
        payload = {
            "customModelId": model_id,
            "customModelVersionId": model_version_id,
            "environmentId": model_info.get_value(
                ModelSchema.VERSION_KEY, ModelSchema.MODEL_ENV_KEY
            ),
        }

        loaded_checks = model_info.get_value(ModelSchema.TEST_KEY, ModelSchema.CHECKS_KEY)
        configuration = self._build_tests_configuration(loaded_checks)
        if configuration:
            payload["configuration"] = configuration

        parameters = self._build_tests_parameters(loaded_checks)
        if parameters:
            payload["parameters"] = parameters

        test_dataset_id = model_info.get_value(ModelSchema.TEST_KEY, ModelSchema.TEST_DATA_KEY)
        if test_dataset_id:  # It may be empty only for unstructured models
            payload["datasetId"] = test_dataset_id

        memory = model_info.get_value(ModelSchema.TEST_KEY, ModelSchema.MEMORY_KEY)
        if memory:
            payload["maximumMemory"] = memory

        response = self._http_requester.post(self.CUSTOM_MODELS_TEST_ROUTE, json=payload)
        if response.status_code != 202:
            raise DataRobotClientError(
                "Custom model version test failed. "
                f"Response code: {response.status_code}. "
                f"Response body: {response.text}."
            )
        return response

    @staticmethod
    def _build_tests_configuration(loaded_checks):
        configuration = {
            "longRunningService": "fail",
            "errorCheck": "fail",
        }
        if loaded_checks:
            for check, info in loaded_checks.items():
                if not info[ModelSchema.CHECK_ENABLED_KEY]:
                    continue

                dr_check_name = DrApiAttrs.to_dr_test_check(check)
                configuration[dr_check_name] = (
                    "fail" if info[ModelSchema.BLOCK_DEPLOYMENT_IF_FAILS_KEY] else "warn"
                )
        return configuration

    @staticmethod
    def _build_tests_parameters(loaded_checks):
        parameters = {}
        if loaded_checks:
            for check, info in loaded_checks.items():
                if not info[ModelSchema.CHECK_ENABLED_KEY]:
                    continue

                check_params = {}
                if check == ModelSchema.PREDICTION_VERIFICATION_KEY:
                    check_params = {
                        "datasetId": info[ModelSchema.OUTPUT_DATASET_KEY],
                        "predictionsColumn": info[ModelSchema.PREDICTIONS_COLUMN],
                    }
                    if ModelSchema.MATCH_THRESHOLD_KEY in info:
                        check_params["matchingRate"] = (
                            info[ModelSchema.PASSING_MATCH_RATE_KEY] / 100
                        )
                    if ModelSchema.PASSING_MATCH_RATE_KEY in info:
                        check_params["comparisonPrecision"] = info[ModelSchema.MATCH_THRESHOLD_KEY]
                elif check == ModelSchema.PERFORMANCE_KEY:
                    if ModelSchema.MAXIMUM_RESPONSE_TIME_KEY in info:
                        check_params["maxResponseTime"] = info[
                            ModelSchema.MAXIMUM_RESPONSE_TIME_KEY
                        ]
                    if ModelSchema.MAXIMUM_EXECUTION_TIME in info:
                        check_params["maxExecutionTime"] = info[ModelSchema.MAXIMUM_EXECUTION_TIME]
                    if ModelSchema.NUMBER_OF_PARALLEL_USERS_KEY in info:
                        check_params["numParallelUsers"] = info[
                            ModelSchema.NUMBER_OF_PARALLEL_USERS_KEY
                        ]
                elif check == ModelSchema.STABILITY_KEY:
                    if ModelSchema.TOTAL_PREDICTION_REQUESTS_KEY in info:
                        check_params["numPredictions"] = info[
                            ModelSchema.TOTAL_PREDICTION_REQUESTS_KEY
                        ]
                    if ModelSchema.PASSING_RATE_KEY in info:
                        check_params["passingRate"] = info[ModelSchema.PASSING_RATE_KEY]
                    if ModelSchema.NUMBER_OF_PARALLEL_USERS_KEY in info:
                        check_params["numParallelUsers"] = info[
                            ModelSchema.NUMBER_OF_PARALLEL_USERS_KEY
                        ]
                    if ModelSchema.MINIMUM_PAYLOAD_SIZE_KEY in info:
                        check_params["minPayloadSize"] = info[ModelSchema.MINIMUM_PAYLOAD_SIZE_KEY]
                    if ModelSchema.MAXIMUM_PAYLOAD_SIZE_KEY in info:
                        check_params["maxPayloadSize"] = info[ModelSchema.MAXIMUM_PAYLOAD_SIZE_KEY]

                if check_params:
                    dr_check_name = DrApiAttrs.to_dr_test_check(check)
                    parameters[dr_check_name] = check_params
        return parameters

    def upload_dataset(self, dataset_filepath):
        """
        Upload a dataset to DataRobot catalogue.

        Parameters
        ----------
        dataset_filepath : str or pathlib.Path
            A local filepath to a dataset.

        Returns
        -------
        str,
            A DataRobot dataset ID.
        """

        with open(dataset_filepath, "rb") as dataset_file:
            data = {"file": (str(dataset_filepath), dataset_file)}
            mp = MultipartEncoder(fields=data)
            headers = {"Content-Type": mp.content_type}
            response = self._http_requester.post(
                self.DATASET_UPLOAD_ROUTE, data=mp, headers=headers
            )
            if response.status_code != 202:
                raise DataRobotClientError(f"Failed uploading dataset. Response: {response.text}")
        location = response.headers["Location"]
        resource = self._wait_for_async_resolution(location)
        dataset_id = resource.split("/")[-2]
        logger.debug(f"Dataset uploaded successfully (id: {dataset_id})")
        return dataset_id

    def delete_dataset(self, dataset_id):
        """
        Delete a dataset from DataRobot catalogue, given a DataRobot dataset ID.

        Parameters
        ----------
        dataset_id : str
            A DataRobot dataset ID.
        """

        response = self._http_requester.delete(f"{self.DATASETS_ROUTE}{dataset_id}/")
        if response.status_code != 204:
            raise DataRobotClientError(f"Failed deleting dataset id '{dataset_id}'")

    def fetch_custom_model_deployments(self, model_ids):
        """
        Retrieve deployments from DataRobot, given a list of DataRobot model IDs.

        Parameters
        ----------
        model_ids : list[str]
            A list of DataRobot model IDs.

        Returns
        -------
        list[dict]
            A list of DataRobot deployments.
        """

        logger.debug(f"Fetching custom model deployments for model ids: '{model_ids}' ...")

        return self._paginated_fetch(
            self.CUSTOM_MODEL_DEPLOYMENTS_ROUTE, json={"customModelIds": model_ids}
        )

    def fetch_deployments(self):
        """
        Retrieve deployments from DataRobot.

        Returns
        -------
        list[dict]
            A list of DataRobot deployments.
        """

        logger.debug("Fetching deployments...")
        return self._paginated_fetch(self.DEPLOYMENTS_ROUTE)

    def fetch_deployment_by_git_id(self, git_deployment_id):
        """
        Retrieve a deployment from DataRobot, given Git deployment ID.

        Parameters
        ----------
        git_deployment_id : str
            A unique ID that is defined by the user.

        Returns
        -------
        dict or None,
            A DataRobot deployment if found, otherwise None.
        """

        deployments = self.fetch_deployments()
        try:
            return next(d for d in deployments if d.get("gitDeploymentId") == git_deployment_id)
        except StopIteration:
            return None

    def create_deployment(self, custom_model_version, deployment_info):
        """
        Create a deployment in DataRobot from a DataRobot custom model version.

        Parameters
        ----------
        custom_model_version : dict
            A DataRobot custom model version.
        deployment_info : DeploymentInfo
            An information about a deployment, which was read from the local source tree.

        Returns
        -------
        dict,
            A DataRobot deployment.
        """

        model_package = self._create_model_package_from_custom_model_version(
            custom_model_version["id"]
        )
        deployment_id = self._create_deployment_from_model_package(model_package, deployment_info)
        deployment = self.update_deployment_settings(deployment_id, deployment_info)
        return deployment

    def _create_model_package_from_custom_model_version(self, custom_model_version_id):
        payload = {"customModelVersionId": custom_model_version_id}
        response = self._http_requester.post(self.MODEL_PACKAGES_CREATE_ROUTE, json=payload)
        if response.status_code != 201:
            raise DataRobotClientError(
                "Failed creating model package from custom model version. "
                f"custom model version id: {custom_model_version_id}, "
                f"Response status: {response.status_code}, "
                f"Response body: {response.text}",
                code=response.status_code,
            )
        return response.json()

    def _create_deployment_from_model_package(self, model_package, deployment_info):
        label = deployment_info.get_settings_value(DeploymentSchema.LABEL_KEY)
        if not label:
            label = f"{model_package['target']['name']} Predictions [GitHub CI/CD]"

        payload = {
            "gitDeploymentId": deployment_info.git_deployment_id,
            "modelPackageId": model_package["id"],
            "label": label,
            "predictionEnvironmentId": self._get_prediction_environment_id(
                model_package, deployment_info
            ),
        }

        importance = deployment_info.get_settings_value(DeploymentSchema.IMPORTANCE_KEY)
        if importance:
            payload["importance"] = importance

        response = self._http_requester.post(self.DEPLOYMENTS_CREATE_ROUTE, json=payload)
        if response.status_code != 202:
            raise DataRobotClientError(
                "Failed creating a deployment from model package."
                f"Git deployment id: {deployment_info.git_deployment_id}, "
                f"Model package id: {model_package['id']}, "
                f"Response status: {response.status_code}, "
                f"Response body: {response.text}",
                code=response.status_code,
            )
        return response.json()["id"]

    def _get_prediction_environment_id(self, model_package, deployment_info):
        prediction_environment_name = deployment_info.get_value(
            DeploymentSchema.PREDICTION_ENVIRONMENT_NAME_KEY
        )
        prediction_envs = self._fetch_prediction_environments(prediction_environment_name)
        if not prediction_envs:
            raise DataRobotClientError(
                "Prediction environment is missing. "
                "Make sure to setup at least one valid prediction environment. "
                f"Git deployment id: {deployment_info.git_deployment_id}, "
                f"Model package id: {model_package['id']}."
            )
        return prediction_envs[0]["id"]

    def update_deployment_settings(self, deployment_id, deployment_info, actual_settings=None):
        """
        This method updates the deployment setting. It can be called with the actual deployment
        settings in order to avoid submission of unneeded settings. The reason for not always
        submitting the desired settings is because a change in a given setting might result
        in a long, heavy computation jobs in the backend. One fundamental rule is that if
        the corresponding definition does not exist in the local definition, it'll not be
        submitted to DataRobot.

        Parameters
        ----------
        deployment_id : str
            The DataRobot deployment ID.
        deployment_info :  DeploymentInfo
            An information about a deployment, which was read from the local source tree.
        actual_settings : dict
            Optional. The settings that were fetched from DataRobot.

        Returns
        -------
        dict,
            The updated deployment from DataRobot.
        """

        payload = {}

        desired_association_section = deployment_info.get_settings_value(
            DeploymentSchema.ASSOCIATION_KEY
        )
        if desired_association_section:
            payload["associationId"] = self._setup_association_payload(
                deployment_info, actual_settings
            )

        desired_target_drift = deployment_info.get_settings_value(
            DeploymentSchema.ENABLE_TARGET_DRIFT_KEY
        )
        if desired_target_drift is not None:
            actual_targe_drift = actual_settings["targetDrift"] if actual_settings else None
            if actual_targe_drift != desired_target_drift:
                payload["targetDrift"] = {"enabled": desired_target_drift}

        desired_feature_drift = deployment_info.get_settings_value(
            DeploymentSchema.ENABLE_FEATURE_DRIFT_KEY
        )
        if desired_feature_drift is not None:
            actual_feature_drift = actual_settings["featureDrift"] if actual_settings else None
            if actual_feature_drift != desired_feature_drift:
                payload["featureDrift"] = {"enabled": desired_feature_drift}

        desired_segmented_analysis = deployment_info.get_settings_value(
            DeploymentSchema.SEGMENT_ANALYSIS_KEY
        )
        if desired_segmented_analysis:
            payload["segmentAnalysis"] = self._setup_segmented_analysis(
                deployment_info, actual_settings
            )

        # A special case, in which the default is to enable challengers
        if deployment_info.is_challenger_enabled:
            payload["predictionsDataCollection"] = {"enabled": True}
        else:
            desired_pred_collection_enabled = deployment_info.get_settings_value(
                DeploymentSchema.ENABLE_PREDICTIONS_COLLECTION_KEY,
            )
            if desired_pred_collection_enabled is not None:
                payload["predictionsDataCollection"] = {"enabled": desired_pred_collection_enabled}

        payload["challengerModels"] = {"enabled": deployment_info.is_challenger_enabled}

        response = self._http_requester.patch(
            self.DEPLOYMENT_SETTINGS_ROUTE.format(deployment_id=deployment_id), json=payload
        )
        if response.status_code != 202:
            raise DataRobotClientError(
                "Failed to update deployment settings."
                f"Git deployment id: {deployment_info.git_deployment_id}, "
                f"Deployment id: {deployment_id}, "
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
                code=response.status_code,
            )
        location = self._wait_for_async_resolution(response.headers["Location"])
        response = self._http_requester.get(location, raw=True)
        return response.json()

    @staticmethod
    def _setup_association_payload(deployment_info, actual_settings):
        association_payload = {}
        desired_association_pred_id = deployment_info.get_settings_value(
            DeploymentSchema.ASSOCIATION_KEY, DeploymentSchema.ASSOCIATION_PRED_ID_KEY
        )
        if desired_association_pred_id is not None:
            actuals_cols = (
                actual_settings["associationId"]["columnNames"] if actual_settings else None
            )
            desired_association_pred_id = [desired_association_pred_id]
            if desired_association_pred_id != actuals_cols:
                association_payload["columnNames"] = desired_association_pred_id

        desired_required = deployment_info.get_settings_value(
            DeploymentSchema.ASSOCIATION_KEY,
            DeploymentSchema.ASSOCIATION_REQUIRED_IN_PRED_REQUEST_KEY,
        )
        if desired_required is not None:
            actual_required = (
                actual_settings["associationId"]["requiredInPredictionRequests"]
                if actual_settings
                else None
            )
            if desired_required != actual_required:
                # NOTE: this is a simplified alternative, which supports a single association ID
                association_payload["requiredInPredictionRequests"] = desired_required

        return association_payload

    @staticmethod
    def _setup_segmented_analysis(deployment_info, actual_settings):
        segmented_analysis_payload = {}
        desired_enabled = deployment_info.get_settings_value(
            DeploymentSchema.SEGMENT_ANALYSIS_KEY,
            DeploymentSchema.ENABLE_SEGMENT_ANALYSIS_KEY,
        )
        if desired_enabled is not None:
            actual_enabled = (
                actual_settings["segmentAnalysis"]["enabled"] if actual_settings else None
            )
            if desired_enabled != actual_enabled:
                segmented_analysis_payload = {"enabled": desired_enabled}

        desired_attributes = deployment_info.get_settings_value(
            DeploymentSchema.SEGMENT_ANALYSIS_KEY,
            DeploymentSchema.SEGMENT_ANALYSIS_ATTRIBUTES_KEY,
        )
        if desired_attributes is not None:
            actual_attributes = (
                actual_settings["segmentAnalysis"]["attributes"] if actual_settings else None
            )
            if desired_attributes != actual_attributes:
                segmented_analysis_payload["attributes"] = desired_attributes

        return segmented_analysis_payload

    def fetch_deployment_settings(self, deployment_id, deployment_info):
        """
        Retrieve deployment settings from DataRobot, given a deployment ID.

        Parameters
        ----------
        deployment_id : str
            A DataRobot deployment ID.
        deployment_info : DeploymentInfo
            An information about a deployment, which was read from the local source tree.

        Returns
        -------
        dict,
            DataRobot deployment settings.
        """

        response = self._http_requester.get(
            self.DEPLOYMENT_SETTINGS_ROUTE.format(deployment_id=deployment_id)
        )
        if response.status_code != 200:
            raise DataRobotClientError(
                "Failed to fetch deployment settings."
                f"Git deployment id: {deployment_info.git_deployment_id}, "
                f"Deployment id: {deployment_id}, "
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
                code=response.status_code,
            )
        return response.json()

    def submit_deployment_actuals(
        self, target_name, association_id, actuals_dataset_id, datarobot_deployment
    ):
        """
        Set a deployment actuals information in DataRobot.

        Parameters
        ----------
        target_name : str
            The target column name in the Actuals dataset.
        association_id : str
            The column name that is used to associate a prediction with the Actuals.
        actuals_dataset_id : str
            A dataset ID from the DataRobot catalogue.
        datarobot_deployment : dict
            A DataRobot deployment.

        Returns
        -------
        dict,
            The Accuracy entity from DataRobot.
        """

        payload = {
            "datasetId": actuals_dataset_id,
            "actualValueColumn": target_name,
            "associationIdColumn": association_id,
        }
        url = self.DEPLOYMENT_ACTUALS_UPDATE_ROUTE.format(deployment_id=datarobot_deployment["id"])
        response = self._http_requester.post(url, json=payload)
        location = self._wait_for_async_resolution(response.headers["Location"])
        response = self._http_requester.get(location, raw=True)
        return response.json()

    def update_deployment_label(self, deployment_id, label):
        """
        Update a deployment label in DataRobot.

        Parameters
        ----------
        deployment_id : str
            A DataRobot deployment ID.
        label : str
            A label to set.
        """

        response = self._http_requester.patch(
            self.DEPLOYMENT_ROUTE.format(deployment_id=deployment_id), json={"label": label}
        )
        if response.status_code != 204:
            raise DataRobotClientError(
                f"Failed to update deployment label. Error: {response.text}.",
                code=response.status_code,
            )

    def delete_deployment_by_id(self, deployment_id):
        """
        Delete a deployment in DataRobot, given a deployment ID.

        Parameters
        ----------
        deployment_id : str
            A DataRobot deployment ID.
        """

        sub_path = f"{self.DEPLOYMENTS_ROUTE}{deployment_id}/"
        response = self._http_requester.delete(sub_path)
        if response.status_code != 204:
            raise DataRobotClientError(
                f"Failed to delete deployment. Error: {response.text}.",
                code=response.status_code,
            )

    def delete_deployment_by_git_id(self, git_deployment_id):
        """
        Delete a deployment from DataRobot, given a Git deployment ID.

        Parameters
        ----------
        git_deployment_id : str
            A unique ID that is defined by the user.
        """

        deployments = self.fetch_deployments()
        try:
            test_deployment = next(
                d for d in deployments if d.get("gitDeploymentId") == git_deployment_id
            )
        except StopIteration:
            raise IllegalModelDeletion(
                f"Given deployment does not exist. git_deployment_id: {git_deployment_id}."
            )
        self.delete_deployment_by_id(test_deployment["id"])

    def _fetch_prediction_environments(self, name=None):
        url = self.PREDICTION_ENVIRONMENTS_ROUTE
        if name:
            url = url + f"&search={name}"

        return self._paginated_fetch(url)

    def replace_model_deployment(self, custom_model_version, datarobot_deployment):
        """
        Replace a custom model version in a given deployment in DataRobot.

        Parameters
        ----------
        custom_model_version : dict
            A DataRobot custom model version.
        datarobot_deployment : dict
            A DataRobot deployment.

        Returns
        -------
        dict,
            A DataRobot deployment, in which the model was replaced.
        """

        model_package = self._create_model_package_from_custom_model_version(
            custom_model_version["id"]
        )
        self._validate_model_compatibility(
            model_package["id"], datarobot_deployment.deployment["id"]
        )
        return self._replace_deployment_model(
            model_package["id"], datarobot_deployment.deployment["id"]
        )

    def _validate_model_compatibility(self, model_package_id, deployment_id):
        payload = {"modelPackageId": model_package_id}
        url = self.DEPLOYMENT_MODEL_VALIDATION_ROUTE.format(deployment_id=deployment_id)
        response = self._http_requester.post(url, json=payload)
        if response.status_code != 200:
            raise DataRobotClientError(
                "A deployment's model validation was failed. "
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
            )

        validation_response = response.json()
        validation_status = validation_response["status"]
        validation_message = validation_response["message"]
        if validation_status == "failing":
            raise DataRobotClientError(validation_message)
        elif validation_status == "warning":
            logger.warning(validation_message)
        else:
            logger.info(validation_message)

    def _replace_deployment_model(self, model_package_id, deployment_id):
        payload = {"modelPackageId": model_package_id, "reason": "DATA_DRIFT"}
        url = self.DEPLOYMENT_MODEL_ROUTE.format(deployment_id=deployment_id)
        response = self._http_requester.patch(url, json=payload)
        if response.status_code != 202:
            raise DataRobotClientError(
                "Failed to replace a model in a deployment."
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
            )
        location = self._wait_for_async_resolution(response.headers["Location"])
        response = self._http_requester.get(location, raw=True)
        deployment = response.json()
        return deployment

    def create_challenger(self, custom_model_version, datarobot_deployment, deployment_info):
        """
        Create a model challenger in DataRobot.

        Parameters
        ----------
        custom_model_version : dict
            A DataRobot custom model version to challenge the existing one.
        datarobot_deployment : dict
            A DataRobot deployment to add the challenger to.
        deployment_info : DeploymentInfo
            An information about the deployment, which is read from the local source tree.

        Returns
        -------
        dict,
            A DataRobot challenger.
        """

        model_package = self._create_model_package_from_custom_model_version(
            custom_model_version["id"]
        )
        deployment_id = datarobot_deployment.deployment["id"]
        self._validate_model_compatibility(model_package["id"], deployment_id)
        return self._create_challenger(model_package, deployment_id, deployment_info)

    def _create_challenger(self, model_package, deployment_id, deployment_info):
        payload = {
            "modelPackageId": model_package["id"],
            "name": model_package["name"],
            "predictionEnvironmentId": self._get_prediction_environment_id(
                model_package, deployment_info
            ),
        }

        url = self.DEPLOYMENT_MODEL_CHALLENGER_ROUTE.format(deployment_id=deployment_id)
        response = self._http_requester.post(url, json=payload)
        if response.status_code != 202:
            raise DataRobotClientError(
                "Failed to submit a challenger."
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
            )
        location = self._wait_for_async_resolution(response.headers["Location"])
        response = self._http_requester.get(location, raw=True)
        return response.json()

    def fetch_challengers(self, deployment_id):
        """
        Retrieve challengers of a given deployment in DataRobot.

        Parameters
        ----------
        deployment_id : str
            A DataRobot deployment ID.

        Returns
        -------
        list[dict],
            A list of challengers of the given deployment in DataRobot.
        """

        url = self.DEPLOYMENT_MODEL_CHALLENGER_ROUTE.format(deployment_id=deployment_id)
        response = self._http_requester.get(url)
        if response.status_code != 200:
            raise DataRobotClientError(
                "Failed to fetch deployment challengers. "
                f"deployment_id: {deployment_id}, "
                f"Response status: {response.status_code}, "
                f"Response body: {response.text}",
            )
        return response.json()["data"]

    def update_training_and_holdout_datasets_for_unstructured_models(
        self, datarobot_custom_model, model_info
    ):
        """
        Update training and holdout datasets of unstructured model in DataRobot.

        Parameters
        ----------
        datarobot_custom_model : dict
            A DataRobot custom model.
        model_info : ModelInfo
            An information about a model, which is read from the local source tree.

        Returns
        -------
        dict or None,
            A custom model entity from DataRobot if an update took place or None otherwise.
        """

        ext_stats_payload = {}
        remote_settings = datarobot_custom_model.get("externalMlopsStatsConfig", {}) or {}

        DatasetParam = namedtuple("DatasetParam", ["local", "remote"])
        dataset_params = [
            DatasetParam(ModelSchema.TRAINING_DATASET_KEY, "trainingDatasetId"),
            DatasetParam(ModelSchema.HOLDOUT_DATASET_KEY, "holdoutDatasetId"),
        ]
        for dataset_param in dataset_params:
            local_value = model_info.get_settings_value(dataset_param.local)
            if local_value and local_value != remote_settings.get(dataset_param.remote):
                ext_stats_payload[dataset_param.remote] = local_value

        if ext_stats_payload:
            payload = {"externalMlopsStatsConfig": ext_stats_payload}
            url = self.CUSTOM_MODEL_ROUTE.format(model_id=datarobot_custom_model["id"])
            response = self._http_requester.patch(url, json=payload)
            if response.status_code != 200:
                raise DataRobotClientError(
                    "Failed to update training / holdout datasets for unstructured model. "
                    f"Git model ID: {model_info.git_model_id}, "
                    f"DataRobot model ID: {datarobot_custom_model['id']}, "
                    f"Response status: {response.status_code}, "
                    f"Response body: {response.text}",
                )
            return response.json()
        return None

    def update_training_dataset_for_structured_models(self, datarobot_custom_model, model_info):
        """
        Updates a training dataset of a structured model in DataRobot, which may contain a
        partition column.

        Parameters
        ----------
        datarobot_custom_model : dict
            A DataRobot custom model.
        model_info : ModelInfo
            An information about the model, which is read from the local source tree.

        Returns
        -------
        CustomModel or None
            The updated custom model from DataRobot if an update took place, or None otherwise.
        """

        training_dataset_payload = {}

        DatasetParam = namedtuple("DatasetParam", ["local", "remote"])
        dataset_params = [
            DatasetParam(ModelSchema.TRAINING_DATASET_KEY, "datasetId"),
            DatasetParam(ModelSchema.PARTITIONING_COLUMN_KEY, "partitionColumn"),
        ]
        for dataset_param in dataset_params:
            local_value = model_info.get_settings_value(dataset_param.local)
            if local_value and local_value != datarobot_custom_model.get(dataset_param.remote):
                training_dataset_payload[dataset_param.remote] = local_value

        if training_dataset_payload:
            url = self.CUSTOM_MODEL_TRAINING_DATA.format(model_id=datarobot_custom_model["id"])
            response = self._http_requester.patch(url, json=training_dataset_payload)
            if response.status_code != 202:
                raise DataRobotClientError(
                    "Failed to update training dataset for structured model. "
                    f"Git model ID: {model_info.git_model_id}. "
                    f"DataRobot model ID: {datarobot_custom_model['id']}. "
                    f"Response status: {response.status_code}. "
                    f"Response body: {response.text}.",
                )
            location = response.headers["Location"]
            self._wait_for_async_resolution(location)
            response = self._http_requester.get(location, raw=True)
            return response.json()
        return None

    def update_model_settings(self, datarobot_custom_model, model_info):
        """
        Update custom inference model settings in DataRobot.

        Parameters
        ----------
        datarobot_custom_model : dict
            A DataRobot custom model.
        model_info : ModelInfo
            An information about the model, which is read from the local source tree.

        Returns
        -------
        dict or None
            The updated custom model from DatRobot if an update took place, or None otherwise.
        """

        payload = {}

        for local_key, remote_key in self.MODEL_SETTINGS_KEYS_MAP.items():
            local_value = model_info.get_settings_value(local_key)
            if local_value and local_value != datarobot_custom_model[remote_key]:
                payload[remote_key] = local_value

        if payload:
            url = self.CUSTOM_MODEL_ROUTE.format(model_id=datarobot_custom_model["id"])
            response = self._http_requester.patch(url, json=payload)
            if response.status_code != 200:
                raise DataRobotClientError(
                    "Failed to update custom model settings. "
                    f"Git model ID: {model_info.git_model_id}. "
                    f"DataRobot model ID: {datarobot_custom_model['id']}. "
                    f"Response status: {response.status_code}. "
                    f"Response body: {response.text}.",
                )
            return response.json()
        return None
