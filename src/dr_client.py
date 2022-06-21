import json
import logging
import time

import requests
from requests_toolbelt import MultipartEncoder

from common.exceptions import HttpRequesterException, DataRobotClientError
from common.http_requester import HttpRequester
from common.string_util import StringUtil
from schema_validator import ModelSchema

logger = logging.getLogger(__name__)


class DrClient:

    CUSTOM_MODELS_ROUTE = "customModels/"
    CUSTOM_MODELS_VERSION_ROUTE = "customModels/{model_id}/versions/"

    def __init__(self, datarobot_webserver, datarobot_api_token):
        if "v2" not in datarobot_webserver:
            datarobot_webserver = f"{StringUtil.slash_suffix(datarobot_webserver)}api/v2/"

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

    def is_accessible(self):
        logger.info("Check if webserver is accessible ...")
        response = self._http_requester.get(
            f"{self._http_requester.webserver_api_path}/ping", raw=True
        )
        return response.status_code == 200 and response.json()["response"] == "pong"

    def fetch_custom_models(self):
        logger.info("Fetching custom models...")
        return self._paginated_fetch(self.CUSTOM_MODELS_ROUTE)

    def _paginated_fetch(self, route_url, **kwargs):
        def _fetch_single_page(url, raw):
            response = self._http_requester.get(url, raw, **kwargs)
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
            "isUnstructuredModelKind": model_info.is_unstructured,
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
        logger.info(f"Fetching custom model versions for model '{custom_model_id}' ...")
        return self._paginated_fetch(
            self.CUSTOM_MODELS_VERSION_ROUTE.format(model_id=custom_model_id), **kwargs
        )

    def fetch_custom_model_latest_version_by_git_model_id(self, git_model_id):
        logger.info(f"Fetching custom model versions for git model '{git_model_id}' ...")

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
                main_branch_commit_sha,
                pull_request_commit_sha,
                changed_files_info,
                file_ids_to_delete=file_ids_to_delete,
                base_env_id=base_env_id,
            )
            mp = MultipartEncoder(fields=payload)
            headers = {"Content-Type": mp.content_type}

            url = self.CUSTOM_MODELS_VERSION_ROUTE.format(model_id=custom_model_id)
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
                f"Failed to delete custom model. Response status: {response.status_code}.",
                code=response.status_code,
            )

    def delete_custom_model_by_git_model_id(self, git_model_id):
        custom_models = self.fetch_custom_models()
        try:
            test_custom_model = next(cm for cm in custom_models if cm["gitModelId"] == git_model_id)
            self.delete_custom_model_by_model_id(test_custom_model["id"])
        except StopIteration:
            raise DataRobotClientError(
                f"Failed to delete custom model. Custom model with '{git_model_id}' "
                f"git model ID was not found."
            )
