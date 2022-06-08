import logging
import time

import responses

from common.exceptions import HttpRequesterException, DataRobotClientError
from common.http_requester import HttpRequester
from schema_validator import ModelSchema

logger = logging.getLogger(__name__)


class DrClient:

    CUSTOM_MODELS_ROUTE = "customModels/"
    CUSTOM_MODELS_VERSION_ROUTE = "customModels/{model_id}/versions/"

    def __init__(self, datarobot_webserver, datarobot_api_token):
        self._http_requester = HttpRequester(datarobot_webserver, datarobot_api_token)

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

    def fetch_custom_models(self):
        logger.info(f"Fetching custom models...")
        return self._paginated_fetch(self.CUSTOM_MODELS_ROUTE)

    def _paginated_fetch(self, route_url):
        def _fetch_single_page(url, raw):
            response = self._http_requester.get(url, raw)
            if response.status_code != 200:
                raise DataRobotClientError(
                    f"Failed to fetch entities of a single page. "
                    f"Response status: {response.status_code} "
                    f"Response body: {response.json()}",
                    code=response.status_code,
                )

            response_json = response.json()
            _total_count = response_json["totalCount"]
            _page_count = response_json["count"]
            _next_page = response_json["next"]
            _returned_models = response_json["data"]
            logger.info(
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
                f"Response body: {response.json()}",
                code=response.status_code,
            )

        custom_model_id = response.json()["id"]
        logger.info(f"Custom model created successfully (ID: {custom_model_id})")
        return custom_model_id

    @staticmethod
    def _setup_payload_for_custom_model_creation(model_info):
        metadata = model_info.metadata
        target_type = ModelSchema.get_value(metadata, ModelSchema.TARGET_TYPE_KEY)

        payload = {
            "customModelType": "inference",  # Currently, there's support only for inference models
            "targetType": target_type,
            "targetName": metadata[ModelSchema.TARGET_NAME_KEY],
            "isUnstructuredKind": model_info.is_unstructured,
            "gitModelId": metadata[ModelSchema.MODEL_ID_KEY],
        }

        name = ModelSchema.get_value(metadata, ModelSchema.SETTINGS_KEY, ModelSchema.NAME_KEY)
        if name:
            payload["name"] = name

        description = ModelSchema.get_value(
            metadata, ModelSchema.SETTINGS_KEY, ModelSchema.DESCRIPTION_KEY
        )
        if description:
            payload["description"] = description

        lang = ModelSchema.get_value(metadata, ModelSchema.LANGUAGE_KEY)
        if lang:
            payload["language"] = lang

        if model_info.is_regression:
            payload["predictionThreshold"] = metadata[ModelSchema.PREDICTION_THRESHOLD_KEY]
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

    def delete_custom_model(self, custom_model_id):
        sub_path = f"{self.CUSTOM_MODELS_ROUTE}/{custom_model_id}/"
        response = self._http_requester.delete(sub_path)
        if response.status_code != 204:
            raise DataRobotClientError(
                f"Failed to delete custom model. Response status: {response.status_code}.",
                code=response.status_code,
            )

    def fetch_custom_model_versions(self, custom_model_id):
        logger.info(f"Fetching custom model versions for model '{custom_model_id}' ...")
        return self._paginated_fetch(
            self.CUSTOM_MODELS_VERSION_ROUTE.format(model_id=custom_model_id)
        )

    def create_custom_model_version(
        self,
        custom_model_id,
        model_info,
        main_branch_commit_sha,
        pull_request_commit_sha=None,
        changed_files=None,
        files_to_delete=None,
    ):
        file_objs = []
        try:
            payload = self._setup_payload_for_custom_model_version_creation(
                model_info,
                main_branch_commit_sha,
                pull_request_commit_sha,
                changed_files,
                files_to_delete,
                file_objs,
            )

            url = self.CUSTOM_MODELS_VERSION_ROUTE.format(model_id=custom_model_id)
            response = self._http_requester.post(url, json=payload)
        finally:
            for file_obj in file_objs:
                file_obj.close()

        if response.status_code != 201:
            raise DataRobotClientError(
                f"Failed to create custom model version. "
                f"Response status: {response.status_code} "
                f"Response body: {response.json()}",
                code=response.status_code,
            )

        version_id = response.json()["id"]
        logger.info(f"Custom model version created successfully (ID: {version_id})")
        return version_id

    @classmethod
    def _setup_payload_for_custom_model_version_creation(
        cls,
        model_info,
        main_branch_commit_sha,
        pull_request_commit_sha,
        changed_file_paths,
        file_path_to_delete,
        file_objs,
    ):
        metadata = model_info.metadata
        payload = {
            "baseEnvironmentId": ModelSchema.get_value(
                metadata, ModelSchema.VERSION_KEY, ModelSchema.MODEL_ENV_KEY
            ),
            "isMajorUpdate": True,
            "gitModelVersion": {
                "mainBranchCommitSha": main_branch_commit_sha,
                "pullRequestCommitSha": pull_request_commit_sha,
            },
        }

        cls._setup_model_version_files(changed_file_paths, file_path_to_delete, payload, file_objs)

        memory = ModelSchema.get_value(metadata, ModelSchema.VERSION_KEY, ModelSchema.MEMORY_KEY)
        if memory:
            payload["maximumMemory"] = memory

        replicas = ModelSchema.get_value(
            metadata, ModelSchema.VERSION_KEY, ModelSchema.REPLICAS_KEY
        )
        if replicas:
            payload["replicas"] = replicas

        return payload

    @staticmethod
    def _setup_model_version_files(changed_file_paths, file_paths_to_delete, payload, file_objs):
        files = []
        file_paths = []
        for file_path in changed_file_paths or []:
            file_path = str(file_path)
            fd = open(file_path, "r")
            file_objs.append(fd)
            files.append((fd, file_path))
            file_paths.append(file_path)

        if files:
            payload["file"] = files
            payload["filePath"] = file_paths

        if file_paths_to_delete:
            payload["filesToDelete"] = [str(fp) for fp in file_paths_to_delete]
