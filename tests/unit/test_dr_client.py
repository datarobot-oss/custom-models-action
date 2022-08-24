#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=protected-access
# pylint: disable=too-many-arguments

"""A module that contains unit-tests for the DataRobot client module."""

import json

import pytest
import responses
from bson import ObjectId

from common.exceptions import DataRobotClientError
from dr_api_attrs import DrApiAttrs
from dr_client import DrClient
from model_file_path import ModelFilePath
from model_info import ModelInfo
from schema_validator import ModelSchema


@pytest.fixture(name="webserver")
def fixture_webserver():
    """A fixture to return a fake DataRobot webserver."""

    return "http://www.datarobot.dummy-app"


@pytest.fixture(name="api_token")
def fixture_api_token():
    """A fixture to return a fake API token."""

    return "123abc"


@pytest.fixture(name="minimal_regression_model_info")
def fixture_minimal_regression_model_info():
    """A fixture to create a ModelInfo with a minimal regression model information."""
    metadata = {
        ModelSchema.MODEL_ID_KEY: "abc123",
        ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION_KEY,
        ModelSchema.SETTINGS_SECTION_KEY: {
            ModelSchema.TARGET_NAME_KEY: "target_column",
            ModelSchema.PREDICTION_THRESHOLD_KEY: 0.5,
        },
        ModelSchema.VERSION_KEY: {
            ModelSchema.MODEL_ENV_ID_KEY: "627790db5621558eedc4c7fa",
        },
    }
    return ModelInfo(
        yaml_filepath="/dummy/yaml/filepath",
        model_path="/dummy/model/path",
        metadata=metadata,
    )


@pytest.fixture(name="regression_model_info")
def fixture_regression_model_info():
    """A fixture to create a local ModelInfo with information of a regression model."""

    metadata = {
        ModelSchema.MODEL_ID_KEY: "abc123",
        ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION_KEY,
        ModelSchema.SETTINGS_SECTION_KEY: {
            ModelSchema.NAME_KEY: "Awesome Model",
            ModelSchema.TARGET_NAME_KEY: "target_column",
            ModelSchema.DESCRIPTION_KEY: "My awesome model",
            ModelSchema.PREDICTION_THRESHOLD_KEY: 0.5,
            ModelSchema.LANGUAGE_KEY: "Python",
        },
        ModelSchema.VERSION_KEY: {
            ModelSchema.MODEL_ENV_ID_KEY: "627790db5621558eedc4c7fa",
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


def mock_paginated_responses(
    total_num_entities, num_entities_in_page, url_factory, entity_response_factory_fn
):
    """A method to mock paginated responses from DataRobot."""

    def _generate_for_single_page(page_index, num_entities, has_next):
        entities_in_page = [
            entity_response_factory_fn(f"id-{page_index}-{index}") for index in range(num_entities)
        ]
        responses.add(
            responses.GET,
            url_factory(page_index),
            json={
                "totalCount": total_num_entities,
                "count": num_entities,
                "next": url_factory(page_index + 1) if has_next else None,
                "data": entities_in_page,
            },
            status=200,
        )
        return entities_in_page

    total_entities = {}
    quotient, remainder = divmod(total_num_entities, num_entities_in_page)
    if quotient == 0 and remainder == 0:
        total_entities[0] = _generate_for_single_page(0, 0, has_next=False)
    else:
        for page in range(quotient):
            has_next = remainder > 0 or page < quotient - 1
            total_entities[page] = _generate_for_single_page(page, num_entities_in_page, has_next)
        if remainder:
            total_entities[quotient] = _generate_for_single_page(
                quotient, remainder, has_next=False
            )
    return total_entities


class TestCustomModelRoutes:
    """Contains cases to test DataRobot custom models routes."""

    @pytest.fixture
    def regression_model_response_factory(self):
        """A factory fixture to generate a regression custom model response."""

        def _inner(model_id):
            return {
                "id": model_id,
                "userProvidedId": f"user-provided-id-{model_id}",
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
        """A fixture to mock a regression custom model response."""

        return regression_model_response_factory("123abc")

    @pytest.fixture
    def custom_models_url_factory(self, paginated_url_factory):
        """A factory fixture to create a paginated page response, given an index."""

        def _inner(page=0):
            return paginated_url_factory(DrClient.CUSTOM_MODELS_ROUTE, page)

        return _inner

    @pytest.fixture
    def custom_models_url(self, custom_models_url_factory):
        """A fixture to return a custom models URL."""

        return custom_models_url_factory(page=0)

    def test_full_payload_setup_for_custom_model_creation(self, regression_model_info):
        """A case to test full payload setup to create a custom model."""

        payload = DrClient._setup_payload_for_custom_model_creation(regression_model_info)
        self._validate_mandatory_attributes_for_regression_model(payload, optional_exist=True)

    @staticmethod
    def _validate_mandatory_attributes_for_regression_model(payload, optional_exist):
        assert "customModelType" in payload
        assert "targetType" in payload
        assert "targetName" in payload
        assert "isUnstructuredModelKind" in payload
        assert "userProvidedId" in payload
        assert "predictionThreshold" in payload
        assert ("name" in payload) == optional_exist
        assert ("description" in payload) == optional_exist
        assert ("language" in payload) == optional_exist

    def test_minimal_payload_setup_for_custom_model_creation(self, minimal_regression_model_info):
        """A case to test a minimal payload setup to create a custom model."""

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
        """A case to test a successful custom model creation."""

        responses.add(responses.POST, custom_models_url, json=regression_model_response, status=201)
        dr_client = DrClient(
            datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False
        )
        custom_model = dr_client.create_custom_model(regression_model_info)
        assert custom_model is not None

    @responses.activate
    def test_create_custom_model_failure(
        self, webserver, api_token, regression_model_info, custom_models_url
    ):
        """A case to test a failure in custom model creation."""

        status_code = 422
        responses.add(responses.POST, custom_models_url, json={}, status=status_code)
        dr_client = DrClient(
            datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False
        )
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
        """A case to test a successful custom model deletion."""

        expected_model = mock_paginated_responses(
            1, 1, custom_models_url_factory, regression_model_response_factory
        )
        expected_model = expected_model[0][0]
        delete_url = f"{custom_models_url}{expected_model['id']}/"
        responses.add(responses.DELETE, delete_url, json={}, status=204)
        dr_client = DrClient(
            datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False
        )
        dr_client.delete_custom_model_by_user_provided_id(expected_model["userProvidedId"])

    @responses.activate
    def test_delete_custom_model_failure(
        self,
        webserver,
        api_token,
        custom_models_url,
        custom_models_url_factory,
        regression_model_response_factory,
    ):
        """A case to test a failure in custom model deletion."""

        expected_model = mock_paginated_responses(
            1, 1, custom_models_url_factory, regression_model_response_factory
        )
        expected_model = expected_model[0][0]
        delete_url = f"{custom_models_url}{expected_model['id']}/"
        status_code = 409
        responses.add(responses.DELETE, delete_url, json={}, status=status_code)
        dr_client = DrClient(
            datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False
        )
        with pytest.raises(DataRobotClientError) as ex:
            dr_client.delete_custom_model_by_user_provided_id(expected_model["userProvidedId"])
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
        """A case to test a successful custom model retrieval."""

        expected_models_in_all_pages = mock_paginated_responses(
            total_num_models,
            num_models_in_page,
            custom_models_url_factory,
            regression_model_response_factory,
        )
        dr_client = DrClient(
            datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False
        )
        total_models_response = dr_client.fetch_custom_models()
        assert len(total_models_response) == total_num_models

        total_expected_models = []
        for models_per_page in expected_models_in_all_pages.values():
            total_expected_models.extend(models_per_page)

        for fetched_model in total_models_response:
            assert fetched_model in total_expected_models


class TestCustomModelVersionRoutes:
    """Contains cases to test DataRobot custom model version routes."""

    @pytest.fixture
    def custom_model_id(self):
        """A fixture to generate a fake custom model ID."""

        return str(ObjectId())

    @pytest.fixture
    def main_branch_commit_sha(self):
        """A fixture to return a dummy main branch commit SHA."""

        return "4e784ec8fa76beebaaf4391f23e0a3f7f666d328"

    @pytest.fixture
    def pull_request_commit_sha(self):
        """A fixture to return a dummy pull request commit SHA."""

        return "4e784ec8fa76beebaaf4391f23e0a3f7f666d329"

    @pytest.fixture
    def ref_name(self):
        """A fixture to return a dummy Git re name."""

        return "feature-branch"

    @pytest.fixture
    def commit_url(self, pull_request_commit_sha):
        """A fixture to return a dummy GitHub commit web URL."""

        return f"https://github.com/user/project/{pull_request_commit_sha}"

    @pytest.fixture
    def regression_model_version_response_factory(
        self, custom_model_id, ref_name, commit_url, main_branch_commit_sha, pull_request_commit_sha
    ):
        """A factory fixture to create a Regression model version response."""

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
                    "ref_name": ref_name,
                    "commit_url": commit_url,
                    "main_branch_commit_sha": main_branch_commit_sha,
                    "pull_request_commit_sha": pull_request_commit_sha,
                },
            }

        return _inner

    @pytest.fixture
    def regression_model_version_response(self, regression_model_version_response_factory):
        """A fixture to return a specific Regression model version with given ID."""

        return regression_model_version_response_factory(str(ObjectId()))

    @pytest.fixture
    def custom_models_version_url_factory(self, custom_model_id, paginated_url_factory):
        """A factory fixture to create a paginated custom model version page, given an index."""

        def _inner(page=0):
            return paginated_url_factory(
                f"{DrClient.CUSTOM_MODELS_VERSIONS_ROUTE}".format(model_id=custom_model_id), page
            )

        return _inner

    def test_full_payload_setup_for_custom_model_version_creation(
        self,
        regression_model_info,
        ref_name,
        commit_url,
        main_branch_commit_sha,
        pull_request_commit_sha,
        single_model_file_paths,
        single_model_root_path,
        workspace_path,
    ):
        """A case to test a full payload setup when creating a custom model version."""

        file_objs = []
        try:
            regression_model_info._model_path = single_model_root_path
            changed_files_info = [
                ModelFilePath(p, regression_model_info.model_path, workspace_path)
                for p in single_model_file_paths
            ]
            payload, file_objs = DrClient._setup_payload_for_custom_model_version_creation(
                regression_model_info,
                ref_name,
                commit_url,
                main_branch_commit_sha,
                pull_request_commit_sha,
                changed_file_paths=changed_files_info,
                file_ids_to_delete=None,
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
        assert git_model_version_json_str, values
        git_model_version_json = json.loads(git_model_version_json_str[0])
        assert "refName" in git_model_version_json
        assert "commitUrl" in git_model_version_json
        assert "mainBranchCommitSha" in git_model_version_json
        assert "pullRequestCommitSha" in git_model_version_json

        if optional_exist:
            assert "file" in keys
            assert "filePath" in keys
            assert "maximumMemory" in keys
            assert "replicas" in keys

    def test_minimal_payload_setup_for_custom_model_version_creation(
        self,
        minimal_regression_model_info,
        ref_name,
        commit_url,
        main_branch_commit_sha,
        pull_request_commit_sha,
    ):
        """A case to test a minimal payload setup when creating a custom model version."""

        payload, file_objs = DrClient._setup_payload_for_custom_model_version_creation(
            minimal_regression_model_info,
            ref_name,
            commit_url,
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
        ref_name,
        commit_url,
        main_branch_commit_sha,
        pull_request_commit_sha,
        regression_model_version_response,
    ):
        """A case to test a successful custom model version creation."""

        url = custom_models_version_url_factory()
        responses.add(responses.POST, url, json=regression_model_version_response, status=201)
        dr_client = DrClient(
            datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False
        )
        version_id = dr_client.create_custom_model_version(
            custom_model_id,
            regression_model_info,
            ref_name,
            commit_url,
            main_branch_commit_sha,
            pull_request_commit_sha,
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
        ref_name,
        commit_url,
        main_branch_commit_sha,
        pull_request_commit_sha,
    ):
        """A case to test a failure in creating a custom model version."""

        status_code = 422
        url = custom_models_version_url_factory()
        responses.add(responses.POST, url, json={}, status=status_code)
        dr_client = DrClient(
            datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False
        )
        with pytest.raises(DataRobotClientError) as ex:
            dr_client.create_custom_model_version(
                custom_model_id,
                regression_model_info,
                ref_name,
                commit_url,
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
        """A case to test a successful retrieval of custom model version."""

        expected_versions_in_all_pages = mock_paginated_responses(
            total_num_model_versions,
            num_model_versions_in_page,
            custom_models_version_url_factory,
            regression_model_version_response_factory,
        )
        dr_client = DrClient(
            datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False
        )
        total_versions_response = dr_client.fetch_custom_model_versions(custom_model_id)
        assert len(total_versions_response) == total_num_model_versions

        total_expected_versions = []
        for versions_per_page in expected_versions_in_all_pages.values():
            total_expected_versions.extend(versions_per_page)

        for fetched_version in total_versions_response:
            assert fetched_version in total_expected_versions

    class TestCustomModelsTestingRoute:
        """Contains unit-tests to test custom model testing routes."""

        @pytest.mark.parametrize("loaded_checks", [None, {}], ids=["none", "empty_dict"])
        def test_minimal_custom_model_testing_configuration(self, loaded_checks):
            """A case to test minimal configuration for a custom model testing."""

            configuration = DrClient._build_tests_configuration(loaded_checks)
            assert configuration == {"longRunningService": "fail", "errorCheck": "fail"}

        @pytest.mark.parametrize("loaded_checks", [None, {}], ids=["none", "empty_dict"])
        def test_minimal_custom_model_testing_parameters(self, loaded_checks):
            """A case to test a minimal number of parameters in custom model testing."""

            parameters = DrClient._build_tests_parameters(loaded_checks)
            assert not parameters

        def test_full_custom_model_testing_configuration(self, mock_full_custom_model_checks):
            """A case to test a full configuration of custom model testing."""

            assert mock_full_custom_model_checks.keys() == DrApiAttrs.DR_TEST_CHECK_MAP.keys()
            configuration = DrClient._build_tests_configuration(mock_full_custom_model_checks)
            for check in DrApiAttrs.DR_TEST_CHECK_MAP:
                assert DrApiAttrs.to_dr_test_check(check) in configuration
            for check in ["longRunningService", "errorCheck"]:
                assert check in configuration

        def test_full_custom_model_testing_configuration_with_all_disabled_checks(
            self, mock_full_custom_model_checks
        ):
            """
            A case to test a full custom model testing configuration, when all the checks are
            disabled.
            """

            assert mock_full_custom_model_checks.keys() == DrApiAttrs.DR_TEST_CHECK_MAP.keys()
            for _, info in mock_full_custom_model_checks.items():
                info[ModelSchema.CHECK_ENABLED_KEY] = False
            configuration = DrClient._build_tests_configuration(mock_full_custom_model_checks)
            assert configuration == {"longRunningService": "fail", "errorCheck": "fail"}

        def test_full_custom_model_testing_parameters(self, mock_full_custom_model_checks):
            """A case to test a full number of parameters in custom model testing."""

            assert mock_full_custom_model_checks.keys() == DrApiAttrs.DR_TEST_CHECK_MAP.keys()
            parameters = DrClient._build_tests_parameters(mock_full_custom_model_checks)
            for check in [
                ModelSchema.PREDICTION_VERIFICATION_KEY,
                ModelSchema.PERFORMANCE_KEY,
                ModelSchema.STABILITY_KEY,
            ]:
                assert DrApiAttrs.to_dr_test_check(check) in parameters

        def test_full_custom_model_testing_parameters_with_all_disabled_checks(
            self, mock_full_custom_model_checks
        ):
            """
            A case to test a full number of testing parameters, when all the checks are disabled.
            """

            assert mock_full_custom_model_checks.keys() == DrApiAttrs.DR_TEST_CHECK_MAP.keys()
            for _, info in mock_full_custom_model_checks.items():
                info[ModelSchema.CHECK_ENABLED_KEY] = False
            parameters = DrClient._build_tests_parameters(mock_full_custom_model_checks)
            assert not parameters


class TestDeploymentRoutes:
    """Contains unit-tests to test the DataRobot deployment routes."""

    @pytest.fixture
    def deployment_response_factory(self):
        """A factory fixture to create a deployment response."""

        def _inner(deployment_id):
            return {"id": deployment_id, "userProvidedId": f"user-provided-id-{deployment_id}"}

        return _inner

    @pytest.fixture
    def deployments_url_factory(self, paginated_url_factory):
        """A factory fixture to create a paginated deployments URLs."""

        def _inner(page=0):
            return paginated_url_factory(DrClient.DEPLOYMENTS_ROUTE, page)

        return _inner

    @pytest.mark.parametrize(
        "total_num_deployments, num_deployments_in_page",
        [(0, 3), (2, 3), (2, 2), (4, 2), (4, 3)],
        ids=[
            "no-models",
            "page-bigger-than-total-deployments",
            "page-equal-total-deployments",
            "page-lower-than-total-deployments-no-remainder",
            "page-lower-than-total-deployments-with-remainder",
        ],
    )
    @responses.activate
    def test_fetch_deployments_success(
        self,
        webserver,
        api_token,
        total_num_deployments,
        num_deployments_in_page,
        deployments_url_factory,
        deployment_response_factory,
    ):
        """A case to test a successful deployments retrieval."""

        expected_deployments_in_all_pages = mock_paginated_responses(
            total_num_deployments,
            num_deployments_in_page,
            deployments_url_factory,
            deployment_response_factory,
        )
        dr_client = DrClient(
            datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False
        )
        total_deployments_response = dr_client.fetch_deployments()
        assert len(total_deployments_response) == total_num_deployments

        total_expected_deployments = []
        for deployments_per_page in expected_deployments_in_all_pages.values():
            total_expected_deployments.extend(deployments_per_page)

        for fetched_deployment in total_deployments_response:
            assert fetched_deployment in total_expected_deployments
