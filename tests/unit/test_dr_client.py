import pytest
import responses
from bson import ObjectId

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
    def regression_model_response(self):
        return {
            "id": "123abc",
            "custom_model_type": "inference",
            "supports_binary_classification": False,
            "supports_regression": True,
            "supports_anomaly_detection": False,
            "target_type": "Regression",
            "prediction_threshold": 0.5,
        }

    @pytest.fixture
    def custom_models_url(self, webserver):
        return f"{webserver}/{DrClient.CUSTOM_MODELS_ROUTE}"

    def test_full_payload_setup_for_custom_model_creation(self, regression_model_info):
        payload = DrClient._setup_payload_for_custom_model_creation(regression_model_info)
        self._validate_mandatory_attributes_for_regression_model(payload, optional_exist=True)

    @staticmethod
    def _validate_mandatory_attributes_for_regression_model(payload, optional_exist):
        assert "customModelType" in payload
        assert "targetType" in payload
        assert "targetName" in payload
        assert "isUnstructuredKind" in payload
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
    def test_delete_custom_model_success(self, webserver, api_token, custom_models_url):
        custom_model_id = str(ObjectId())
        delete_url = f"{custom_models_url}/{custom_model_id}/"
        responses.add(responses.DELETE, delete_url, json={}, status=204)
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        dr_client.delete_custom_model(custom_model_id)

    @responses.activate
    def test_delete_custom_model_failure(self, webserver, api_token, custom_models_url):
        custom_model_id = str(ObjectId())
        delete_url = f"{custom_models_url}/{custom_model_id}/"
        status_code = 409
        responses.add(responses.DELETE, delete_url, json={}, status=status_code)
        dr_client = DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token)
        with pytest.raises(DataRobotClientError) as ex:
            dr_client.delete_custom_model(custom_model_id)
        assert ex.value.code == status_code


class TestCustomModelVersionRoutes:
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
    def regression_model_version_response(self, main_branch_commit_sha, pull_request_commit_sha):
        return {
            "id": "456def",
            "custom_model_id": "123abc",
            "created": "2022-1-06 13:36:00",
            "items": [
                ("custom.py", "629741dc5621557833bd5aa1"),
                ("util/helper.py", "629741dc5621557833bd5aa2"),
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

    @pytest.fixture
    def custom_models_version_url_factory(self, webserver):
        def _inner(custom_model_id):
            return f"{webserver}/{DrClient.CUSTOM_MODELS_VERSION_ROUTE}".format(
                model_id=custom_model_id
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
            payload = DrClient._setup_payload_for_custom_model_version_creation(
                regression_model_info,
                main_branch_commit_sha,
                pull_request_commit_sha,
                changed_file_paths=single_model_file_paths,
                file_path_to_delete=["README.md"],
                file_objs=file_objs,
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
        assert "baseEnvironmentId" in payload
        assert "isMajorUpdate" in payload

        assert "gitModelVersion" in payload
        assert "mainBranchCommitSha" in payload["gitModelVersion"]
        assert "pullRequestCommitSha" in payload["gitModelVersion"]

        if optional_exist:
            assert "file" in payload
            assert "filePath" in payload
            assert "filesToDelete" in payload
            assert "maximumMemory" in payload
            assert "replicas" in payload

    def test_minimal_payload_setup_for_custom_model_version_creation(
        self, minimal_regression_model_info, main_branch_commit_sha, pull_request_commit_sha
    ):
        file_objs = []
        payload = DrClient._setup_payload_for_custom_model_version_creation(
            minimal_regression_model_info,
            main_branch_commit_sha,
            pull_request_commit_sha,
            None,
            None,
            file_objs,
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
        regression_model_info,
        custom_models_version_url_factory,
        main_branch_commit_sha,
        pull_request_commit_sha,
        regression_model_version_response,
    ):
        custom_model_id = str(ObjectId())
        url = custom_models_version_url_factory(custom_model_id)
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
        regression_model_info,
        custom_models_version_url_factory,
        main_branch_commit_sha,
        pull_request_commit_sha,
        regression_model_version_response,
    ):
        status_code = 422
        custom_model_id = str(ObjectId())
        url = custom_models_version_url_factory(custom_model_id)
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
