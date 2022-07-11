import json
import logging
import time

from requests_toolbelt import MultipartEncoder

from common.exceptions import DataRobotClientError
from common.exceptions import HttpRequesterException
from common.exceptions import IllegalModelDeletion
from common.http_requester import HttpRequester
from common.string_util import StringUtil
from dr_api_attrs import DrApiAttrs
from schema_validator import ModelSchema

logger = logging.getLogger(__name__)


class DrClient:

    CUSTOM_MODELS_ROUTE = "customModels/"
    CUSTOM_MODELS_VERSIONS_ROUTE = "customModels/{model_id}/versions/"
    CUSTOM_MODELS_VERSION_ROUTE = "customModels/{model_id}/versions/{model_ver_id}/"
    CUSTOM_MODELS_TEST_ROUTE = "customModelTests/"
    DATASETS_ROUTE = "datasets/"
    DATASET_UPLOAD_ROUTE = DATASETS_ROUTE + "fromFile/"
    CUSTOM_MODEL_DEPLOYMENTS = "/customModelDeployments/"
    DEPLOYMENTS_ROUTE = "deployments/"

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
        logger.debug("Check if webserver is accessible ...")
        response = self._http_requester.get(
            f"{self._http_requester.webserver_api_path}/ping", raw=True
        )
        return response.status_code == 200 and response.json()["response"] == "pong"

    def fetch_custom_models(self):
        logger.debug("Fetching custom models...")
        return self._paginated_fetch(self.CUSTOM_MODELS_ROUTE)

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
        payload = self._setup_payload_for_custom_model_creation(model_info)
        response = self._http_requester.post(self.CUSTOM_MODELS_ROUTE, json=payload)
        if response.status_code != 201:
            raise DataRobotClientError(
                f"Failed to create custom model. "
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
                code=response.status_code,
            )

        custom_model_id = response.json()["id"]
        logger.debug(f"Custom model created successfully (ID: {custom_model_id})")
        return custom_model_id

    @staticmethod
    def _setup_payload_for_custom_model_creation(model_info):
        metadata = model_info.metadata
        target_type = ModelSchema.get_value(metadata, ModelSchema.TARGET_TYPE_KEY)

        payload = {
            "customModelType": "inference",  # Currently, there's support only for inference models
            "targetType": target_type,
            "targetName": metadata[ModelSchema.TARGET_NAME_KEY],
            "isUnstructuredModelKind": model_info.is_unstructured,
            "gitModelId": metadata[ModelSchema.MODEL_ID_KEY],
        }

        name = ModelSchema.get_value(
            metadata, ModelSchema.SETTINGS_SECTION_KEY, ModelSchema.NAME_KEY
        )
        if name:
            payload["name"] = name

        description = ModelSchema.get_value(
            metadata, ModelSchema.SETTINGS_SECTION_KEY, ModelSchema.DESCRIPTION_KEY
        )
        if description:
            payload["description"] = description

        lang = ModelSchema.get_value(metadata, ModelSchema.LANGUAGE_KEY)
        if lang:
            payload["language"] = lang

        if model_info.is_regression:
            regression_threshold = ModelSchema.get_value(
                metadata, ModelSchema.PREDICTION_THRESHOLD_KEY
            )
            if regression_threshold is not None:
                payload["predictionThreshold"] = regression_threshold
        elif model_info.is_binary:
            payload.update(
                {
                    "positiveClassLabel": metadata[ModelSchema.POSITIVE_CLASS_LABEL_KEY],
                    "negativeClassLabel": metadata[ModelSchema.NEGATIVE_CLASS_LABEL_KEY],
                }
            )
        elif model_info.is_multiclass:
            payload["classLabels"] = metadata[ModelSchema.CLASS_LABELS_KEY]

        return payload

    def fetch_custom_model_versions(self, custom_model_id, **kwargs):
        logger.debug(f"Fetching custom model versions for model '{custom_model_id}' ...")
        return self._paginated_fetch(
            self.CUSTOM_MODELS_VERSIONS_ROUTE.format(model_id=custom_model_id), **kwargs
        )

    def fetch_custom_model_version(self, custom_model_id, custom_model_version_id):
        logger.debug(
            f"Fetching custom model version '{custom_model_version_id}' "
            f"for model '{custom_model_id}' ..."
        )
        url = self.CUSTOM_MODELS_VERSION_ROUTE.format(
            model_id=custom_model_id, model_ver_id=custom_model_version_id
        )
        response = self._http_requester.get(url)
        if response != 200:
            raise DataRobotClientError(
                f"Failed to get custom model version {custom_model_version_id} "
                f"of model {custom_model_id}. "
                f"Response status: {response.status_code} "
                f"Response body: {response.text}",
                code=response.status_code,
            )
        return response.json()

    def fetch_custom_model_latest_version_by_git_model_id(self, git_model_id):
        logger.debug(f"Fetching custom model versions for git model '{git_model_id}' ...")

        custom_models = self.fetch_custom_models()
        custom_model = next(cm for cm in custom_models if cm.get("gitModelId") == git_model_id)
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
        commit_url,
        main_branch_commit_sha,
        pull_request_commit_sha=None,
        changed_files_info=None,
        file_ids_to_delete=None,
        from_latest=False,
    ):
        file_objs = []
        try:
            base_env_id = ModelSchema.get_value(
                model_info.metadata, ModelSchema.VERSION_KEY, ModelSchema.MODEL_ENV_KEY
            )
            payload, file_objs = self._setup_payload_for_custom_model_version_creation(
                model_info,
                commit_url,
                main_branch_commit_sha,
                pull_request_commit_sha,
                changed_files_info,
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
        logger.info(f"Custom model version created successfully (ID: {version_id})")
        return version_id

    @classmethod
    def _setup_payload_for_custom_model_version_creation(
        cls,
        model_info,
        commit_url,
        main_branch_commit_sha,
        pull_request_commit_sha,
        changed_files_info,
        file_ids_to_delete=None,
        base_env_id=None,
    ):
        metadata = model_info.metadata
        payload = [
            ("isMajorUpdate", str(True)),
            (
                "gitModelVersion",
                json.dumps(
                    {
                        "commitUrl": commit_url,
                        "mainBranchCommitSha": main_branch_commit_sha,
                        "pullRequestCommitSha": pull_request_commit_sha,
                    }
                ),
            ),
        ]

        file_objs = cls._setup_model_version_files(changed_files_info, file_ids_to_delete, payload)

        if base_env_id:
            payload.append(("baseEnvironmentId", base_env_id))

        memory = ModelSchema.get_value(metadata, ModelSchema.VERSION_KEY, ModelSchema.MEMORY_KEY)
        if memory:
            payload.append(("maximumMemory", str(memory)))

        replicas = ModelSchema.get_value(
            metadata, ModelSchema.VERSION_KEY, ModelSchema.REPLICAS_KEY
        )
        if replicas:
            payload.append(("replicas", str(replicas)))

        return payload, file_objs

    @staticmethod
    def _setup_model_version_files(changed_files_info, file_ids_to_delete, payload):
        file_objs = []
        for file_info in changed_files_info or []:
            file_path = str(file_info.actual_path)
            fd = open(file_path, "rb")
            file_objs.append(fd)
            path_under_model = str(file_info.path_under_model)

            payload.append(("file", (path_under_model, fd)))
            payload.append(("filePath", path_under_model))

        for file_id_to_delete in file_ids_to_delete or []:
            payload.append(("filesToDelete", file_id_to_delete))

        return file_objs

    def delete_custom_model_by_model_id(self, custom_model_id):
        sub_path = f"{self.CUSTOM_MODELS_ROUTE}{custom_model_id}/"
        response = self._http_requester.delete(sub_path)
        if response.status_code != 204:
            raise DataRobotClientError(
                f"Failed to delete custom model. Error: {response.text}.",
                code=response.status_code,
            )

    def delete_custom_model_by_git_model_id(self, git_model_id):
        custom_models = self.fetch_custom_models()
        try:
            test_custom_model = next(
                cm for cm in custom_models if cm.get("gitModelId") == git_model_id
            )
        except StopIteration:
            raise IllegalModelDeletion(
                f"Given custom model does not exist. git_model_id: {git_model_id}."
            )
        self.delete_custom_model_by_model_id(test_custom_model["id"])

    def run_custom_model_version_testing(self, model_id, model_version_id, model_info):
        """
        Post a query to start custom model version testing.

        Parameters
        ----------
        model_id : str, ObjectId
            A custom model ID, which is generated by DataRobot.
        model_version_id : str, ObjectId
            A custom model version ID, which is generated by DataRobot.
        model_info : ModelInfo
            A structured that contains full information about a single model.

        Raises
        -------
        DataRobotClientError
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
            f"Custom model testing pass with success. Git model ID: {model_info.git_model_id}"
        )

    def _post_custom_model_test_request(self, model_id, model_version_id, model_info):
        payload = {
            "customModelId": model_id,
            "customModelVersionId": model_version_id,
            "environmentId": ModelSchema.get_value(
                model_info.metadata, ModelSchema.VERSION_KEY, ModelSchema.MODEL_ENV_KEY
            ),
        }

        loaded_checks = ModelSchema.get_value(
            model_info.metadata, ModelSchema.TEST_KEY, ModelSchema.CHECKS_KEY
        )
        configuration = self._build_tests_configuration(loaded_checks)
        if configuration:
            payload["configuration"] = configuration

        parameters = self._build_tests_parameters(loaded_checks)
        if parameters:
            payload["parameters"] = parameters

        test_dataset_id = ModelSchema.get_value(
            model_info.metadata, ModelSchema.TEST_KEY, ModelSchema.TEST_DATA_KEY
        )
        if test_dataset_id:  # It may be empty only for unstructured models
            payload["datasetId"] = test_dataset_id

        memory = ModelSchema.get_value(
            model_info.metadata, ModelSchema.TEST_KEY, ModelSchema.MEMORY_KEY
        )
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
        logger.debug(f"Dataset uploaded successfully (ID: {dataset_id})")
        return dataset_id

    def delete_dataset(self, dataset_id):
        response = self._http_requester.delete(f"{self.DATASETS_ROUTE}{dataset_id}/")
        if response.status_code != 204:
            raise DataRobotClientError(f"Failed deleting dataset ID '{dataset_id}'")

    def fetch_custom_model_deployments(self, model_ids):
        logger.debug(f"Fetching custom model deployments for model ids: '{model_ids}' ...")

        return self._paginated_fetch(
            self.CUSTOM_MODEL_DEPLOYMENTS, json={"customModelIds": model_ids}
        )

    def fetch_deployments(self):
        logger.debug("Fetching deployments...")
        return self._paginated_fetch(self.DEPLOYMENTS_ROUTE)
