import json

import pytest
import responses
from bson import ObjectId

from common.data_types import FileInfo
from common.exceptions import DataRobotClientError
from custom_inference_model import ModelInfo
from dr_client import DrClient
from schema_validator import ModelSchema


@pytest.fixture
def webserver():
    return "http://www.datarobot.dummy-app"


@pytest.fixture
def api_token():
    return "123abc"


def mock_paginated_responses(
    total_num_entities, num_entities_in_page, url_factory, entity_response_factory_fn
):
    def _generate_for_single_page(page_index, num_entities, has_next):
        models_in_page = [
            entity_response_factory_fn(f"id-{page_index}-{index}") for index in range(num_entities)
        ]
        responses.add(
            responses.GET,
            url_factory(page_index),
            json={
                "totalCount": total_num_entities,
                "count": num_entities,
                "next": url_factory(page_index + 1) if has_next else None,
                "data": models_in_page,
            },
            status=200,
        )
        return models_in_page

    total_entities = {}
    quotient, remainder = divmod(total_num_entities, num_entities_in_page)
    for page in range(quotient):
        has_next = remainder > 0 or page < quotient - 1
        total_entities[page] = _generate_for_single_page(page, num_entities_in_page, has_next)
    if remainder:
        total_entities[quotient] = _generate_for_single_page(quotient, remainder, has_next=False)
    return total_entities


class TestCustomModelRoutes:
    @pytest.fixture
    def minimal_regression_model_info(self):
        metadata = {
            ModelSchema.MODEL_ID_KEY: "abc123",
            ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION_KEY,
            ModelSchema.TARGET_NAME_KEY: "target_column",
            ModelSchema.PREDICTION_THRESHOLD_KEY: 0.5,
            ModelSchema.VERSION_KEY: {
                ModelSchema.MODEL_ENV_KEY: "627790db5621558eedc4c7fa",
            },
        }
        return ModelInfo(
            yaml_filepath="/dummy/yaml/filepath",
            model_path="/dummy/model/path",
            metadata=metadata,
        )

    @pytest.fixture
    def regression_model_info(self):
        metadata = {
            ModelSchema.MODEL_ID_KEY: "abc123",
            ModelSchema.DEPLOYMENT_ID_KEY: "edf456",
            ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION_KEY,
            ModelSchema.TARGET_NAME_KEY: "target_column",
            ModelSchema.PREDICTION_THRESHOLD_KEY: 0.5,
            ModelSchema.LANGUAGE_KEY: "Python",
            ModelSchema.SETTINGS_KEY: {
                ModelSchema.NAME_KEY: "Awesome Model",
                ModelSchema.DESCRIPTION_KEY: "My awesome model",
            },
            ModelSchema.VERSION_KEY: {
                ModelSchema.MODEL_ENV_KEY: "627790db5621558eedc4c7fa",
                ModelSchema.INCLUDE_GLOB_KEY: ["./"],
                ModelSchema.EXCLUDE_GLOB_KEY: ["README.md", "out/"],
            },
        }
        return ModelInfo(
            yaml_filepath="/dummy/yaml/filepath",
            model_path="/dummy/model/path",
            metadata=metadata,
        )

    @pytest.fixture
    def regression_model_response_factory(self):
        def _inner(model_id):
            return {
                "id": model_id,
                "gitModelId": f"git-id-{model_id}",
                "customModelType": "inference",
                "supportsBinaryClassification": False,
                "supportsRegression": True,
                "supportsAnomalyDetection": False,
                "targetType": "Regression",
                "predictionThreshold": 0.5,
            }

        return _inner

    @pytest.fixture
    def regression_model_response(self, regression_model_response_factory):
        return regression_model_response_factory("123abc")

    @pytest.fixture
    def custom_models_url_factory(self, paginated_url_factory):
        def _inner(page=0):
            return paginated_url_factory(DrClient.CUSTOM_MODELS_ROUTE, page)

        return _inner

    @pytest.fixture
    def custom_models_url(self, custom_models_url_factory):
        return custom_models_url_factory(page=0)

    def test_full_payload_setup_for_custom_model_creation(self, regression_model_info):
        payload = DrClient._setup_payload_for_custom_model_creation(regression_model_info)
        self._validate_mandatory_attributes_for_regression_model(payload, optional_exist=True)

    @staticmethod
    def _validate_mandatory_attributes_for_regression_model(payload, optional_exist):
        assert "customModelType" in payload
        assert "targetType" in payload
        assert "targetName" in payload
        assert "isUnstructuredModelKind" in payload
        assert "gitModelId" in payload
        assert "predictionThreshold" in payload
        assert ("name" in payload) == optional_exist
        assert ("description" in payload) == optional_exist
        assert ("language" in payload) == optional_exist

    def test_minimal_payload_setup_for_custom_model_creation(self, minimal_regression_model_info):
        payload = DrClient._setup_payload_for_custom_model_creation(minimal_regression_model_info)
        self._validate_mandatory_attributes_for_regression_model(payload, optional_exist=False)

    @responses.activate
    def test_create_custom_model_success(
        self,
        webserver,
        api_token,
        regression_model_info,
        custom_models_url,
        regression_model_response,
    ):
        responses.add(responses.POST, custom_models_url, json=regression_model_response, status=201)
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        custom_model_id = dr_client.create_custom_model(regression_model_info)
        assert custom_model_id is not None

    @responses.activate
    def test_create_custom_model_failure(
        self,
        webserver,
        api_token,
        regression_model_info,
        custom_models_url,
        regression_model_response,
    ):
        status_code = 422
        responses.add(responses.POST, custom_models_url, json={}, status=status_code)
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        with pytest.raises(DataRobotClientError) as ex:
            dr_client.create_custom_model(regression_model_info)
        assert ex.value.code == status_code

    @responses.activate
    def test_delete_custom_model_success(
        self,
        webserver,
        api_token,
        custom_models_url,
        custom_models_url_factory,
        regression_model_response_factory,
    ):
        expected_model = mock_paginated_responses(
            1, 1, custom_models_url_factory, regression_model_response_factory
        )
        expected_model = expected_model[0][0]
        delete_url = f"{custom_models_url}{expected_model['id']}/"
        responses.add(responses.DELETE, delete_url, json={}, status=204)
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        dr_client.delete_custom_model_by_git_model_id(expected_model["gitModelId"])

    @responses.activate
    def test_delete_custom_model_failure(
        self,
        webserver,
        api_token,
        custom_models_url,
        custom_models_url_factory,
        regression_model_response_factory,
    ):
        expected_model = mock_paginated_responses(
            1, 1, custom_models_url_factory, regression_model_response_factory
        )
        expected_model = expected_model[0][0]
        delete_url = f"{custom_models_url}{expected_model['id']}/"
        status_code = 409
        responses.add(responses.DELETE, delete_url, json={}, status=status_code)
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        with pytest.raises(DataRobotClientError) as ex:
            dr_client.delete_custom_model_by_git_model_id(expected_model["gitModelId"])
        assert ex.value.code == status_code

    @pytest.mark.parametrize(
        "total_num_models, num_models_in_page",
        [(2, 3), (2, 2), (4, 2), (4, 3)],
        ids=[
            "page-bigger-than-total-models",
            "page-equal-total-models",
            "page-lower-than-total-models-no-remainder",
            "page-lower-than-total-models-with-remainder",
        ],
    )
    @responses.activate
    def test_fetch_custom_models_success(
        self,
        webserver,
        api_token,
        total_num_models,
        num_models_in_page,
        custom_models_url_factory,
        regression_model_response_factory,
    ):
        expected_models_in_all_pages = mock_paginated_responses(
            total_num_models,
            num_models_in_page,
            custom_models_url_factory,
            regression_model_response_factory,
        )
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        total_models_response = dr_client.fetch_custom_models()
        assert len(total_models_response) == total_num_models

        total_expected_models = []
        for models_per_page in expected_models_in_all_pages.values():
            total_expected_models.extend(models_per_page)

        for fetched_model in total_models_response:
            assert fetched_model in total_expected_models


class TestCustomModelVersionRoutes:
    @pytest.fixture
    def custom_model_id(self):
        return str(ObjectId())

    @pytest.fixture
    def minimal_regression_model_info(self):
        metadata = {
            ModelSchema.MODEL_ID_KEY: "abc123",
            ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION_KEY,
            ModelSchema.TARGET_NAME_KEY: "target_column",
            ModelSchema.PREDICTION_THRESHOLD_KEY: 0.5,
            ModelSchema.VERSION_KEY: {
                ModelSchema.MODEL_ENV_KEY: "627790db5621558eedc4c7fa",
            },
        }
        return ModelInfo(
            yaml_filepath="/dummy/yaml/filepath",
            model_path="/dummy/model/path",
            metadata=metadata,
        )

    @pytest.fixture
    def regression_model_info(self):
        metadata = {
            ModelSchema.MODEL_ID_KEY: "abc123",
            ModelSchema.DEPLOYMENT_ID_KEY: "edf456",
            ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION_KEY,
            ModelSchema.TARGET_NAME_KEY: "target_column",
            ModelSchema.PREDICTION_THRESHOLD_KEY: 0.5,
            ModelSchema.VERSION_KEY: {
                ModelSchema.MODEL_ENV_KEY: "627790db5621558eedc4c7fa",
                ModelSchema.INCLUDE_GLOB_KEY: ["./"],
                ModelSchema.EXCLUDE_GLOB_KEY: ["README.md", "out/"],
                ModelSchema.MEMORY_KEY: 256 * 1024 * 1024,
                ModelSchema.REPLICAS_KEY: 3,
            },
        }
        return ModelInfo(
            yaml_filepath="/dummy/yaml/filepath",
            model_path="/dummy/model/path",
            metadata=metadata,
        )

    @pytest.fixture
    def main_branch_commit_sha(self):
        return "4e784ec8fa76beebaaf4391f23e0a3f7f666d328"

    @pytest.fixture
    def pull_request_commit_sha(self):
        return "4e784ec8fa76beebaaf4391f23e0a3f7f666d329"

    @pytest.fixture
    def regression_model_version_response_factory(
        self, custom_model_id, main_branch_commit_sha, pull_request_commit_sha
    ):
        def _inner(version_id):
            return {
                "id": version_id,
                "custom_model_id": custom_model_id,
                "created": "2022-1-06 13:36:00",
                "items": [
                    {
                        "id": "629741dc5621557833bd5aa1",
                        "file_name": "custom.py",
                        "file_path": "custom.py",
                        "file_source": "s3",
                    },
                    {
                        "id": "629741dc5621557833bd5aa2",
                        "file_name": "util/helper.py",
                        "file_path": "util/helper.py",
                        "file_source": "s3",
                    },
                ],
                "is_frozen": False,
                "version_major": 1,
                "version_minor": 2,
                "label": "1.2",
                "baseEnvironmentId": "629741dc5621557833bd5aa3",
                "git_model_version": {
                    "main_branch_commit_sha": main_branch_commit_sha,
                    "pull_request_commit_sha": pull_request_commit_sha,
                },
            }

        return _inner

    @pytest.fixture
    def regression_model_version_response(self, regression_model_version_response_factory):
        return regression_model_version_response_factory(str(ObjectId()))

    @pytest.fixture
    def custom_models_version_url_factory(self, custom_model_id, paginated_url_factory):
        def _inner(page=0):
            return paginated_url_factory(
                f"{DrClient.CUSTOM_MODELS_VERSION_ROUTE}".format(model_id=custom_model_id), page
            )

        return _inner

    def test_full_payload_setup_for_custom_model_version_creation(
        self,
        regression_model_info,
        main_branch_commit_sha,
        pull_request_commit_sha,
        single_model_file_paths,
    ):
        file_objs = []
        try:
            changed_files_info = [FileInfo(p, p) for p in single_model_file_paths]
            payload, file_objs = DrClient._setup_payload_for_custom_model_version_creation(
                regression_model_info,
                main_branch_commit_sha,
                pull_request_commit_sha,
                changed_files_info=changed_files_info,
                file_path_to_delete=None,
                base_env_id=str(ObjectId()),
            )
            self._validate_mandatory_attributes_for_regression_model_version(
                payload, optional_exist=True
            )
            assert len(single_model_file_paths) == len(file_objs)
        finally:
            for file_obj in file_objs:
                file_obj.close()

    @staticmethod
    def _validate_mandatory_attributes_for_regression_model_version(payload, optional_exist):
        keys, values = zip(*payload)
        assert "baseEnvironmentId" in keys
        assert "isMajorUpdate" in keys

        assert "gitModelVersion" in keys
        git_model_version_json_str = [
            v for v in values if isinstance(v, str) and "mainBranchCommitSha" in v
        ]
        assert git_model_version_json_str, git_model_version_json_str
        git_model_version_json = json.loads(git_model_version_json_str[0])
        assert "mainBranchCommitSha" in git_model_version_json
        assert "pullRequestCommitSha" in git_model_version_json

        if optional_exist:
            assert "file" in keys
            assert "filePath" in keys
            assert "maximumMemory" in keys
            assert "replicas" in keys

    def test_minimal_payload_setup_for_custom_model_version_creation(
        self, minimal_regression_model_info, main_branch_commit_sha, pull_request_commit_sha
    ):
        payload, file_objs = DrClient._setup_payload_for_custom_model_version_creation(
            minimal_regression_model_info,
            main_branch_commit_sha,
            pull_request_commit_sha,
            None,
            None,
            base_env_id=str(ObjectId()),
        )
        self._validate_mandatory_attributes_for_regression_model_version(
            payload, optional_exist=False
        )
        assert not file_objs

    @responses.activate
    def test_create_custom_model_version_success(
        self,
        webserver,
        api_token,
        custom_model_id,
        regression_model_info,
        custom_models_version_url_factory,
        main_branch_commit_sha,
        pull_request_commit_sha,
        regression_model_version_response,
    ):
        url = custom_models_version_url_factory()
        responses.add(responses.POST, url, json=regression_model_version_response, status=201)
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        version_id = dr_client.create_custom_model_version(
            custom_model_id, regression_model_info, main_branch_commit_sha, pull_request_commit_sha
        )
        assert version_id is not None

    @responses.activate
    def test_create_custom_model_version_failure(
        self,
        webserver,
        api_token,
        custom_model_id,
        regression_model_info,
        custom_models_version_url_factory,
        main_branch_commit_sha,
        pull_request_commit_sha,
        regression_model_version_response,
    ):
        status_code = 422
        url = custom_models_version_url_factory()
        responses.add(responses.POST, url, json={}, status=status_code)
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        with pytest.raises(DataRobotClientError) as ex:
            dr_client.create_custom_model_version(
                custom_model_id,
                regression_model_info,
                main_branch_commit_sha,
                pull_request_commit_sha,
            )
        assert ex.value.code == status_code

    @pytest.mark.parametrize(
        "total_num_model_versions, num_model_versions_in_page",
        [(2, 3), (2, 2), (4, 2), (4, 3)],
        ids=[
            "page-bigger-than-total-models",
            "page-equal-total-models",
            "page-lower-than-total-models-no-remainder",
            "page-lower-than-total-models-with-remainder",
        ],
    )
    @responses.activate
    def test_fetch_custom_model_versions_success(
        self,
        webserver,
        api_token,
        custom_model_id,
        total_num_model_versions,
        num_model_versions_in_page,
        custom_models_version_url_factory,
        regression_model_version_response_factory,
    ):
        expected_versions_in_all_pages = mock_paginated_responses(
            total_num_model_versions,
            num_model_versions_in_page,
            custom_models_version_url_factory,
            regression_model_version_response_factory,
        )
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        total_versions_response = dr_client.fetch_custom_model_versions(custom_model_id)
        assert len(total_versions_response) == total_num_model_versions

        total_expected_versions = []
        for versions_per_page in expected_versions_in_all_pages.values():
            total_expected_versions.extend(versions_per_page)

        for fetched_version in total_versions_response:
            assert fetched_version in total_expected_versions
