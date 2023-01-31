#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=protected-access
# pylint: disable=too-many-arguments
# pylint: disable=too-many-lines

"""A module that contains unit-tests for the DataRobot client module."""

import contextlib
import json
import logging

import pytest
import responses
from bson import ObjectId
from mock import Mock
from mock import patch

from common.exceptions import DataRobotClientError
from common.http_requester import HttpRequester
from deployment_info import DeploymentInfo
from dr_api_attrs import DrApiAttrs
from dr_client import DrClient
from dr_client import logger as dr_client_logger
from model_file_path import ModelFilePath
from model_info import ModelInfo
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema


@pytest.fixture(name="webserver")
def fixture_webserver():
    """A fixture to return a fake DataRobot webserver."""

    return "http://www.datarobot.dummy-app"


@pytest.fixture(name="api_token")
def fixture_api_token():
    """A fixture to return a fake API token."""

    return "123abc"


@pytest.fixture(name="dr_client")
def fixture_dr_client(webserver, api_token):
    """A fixture to create the DataRobot client."""

    return DrClient(datarobot_webserver=webserver, datarobot_api_token=api_token, verify_cert=False)


@pytest.fixture(name="custom_model_id")
def fixture_custom_model_id():
    """A fixture to generate a fake custom model ID."""

    return str(ObjectId())


@pytest.fixture(name="custom_model_version_id")
def fixture_custom_model_version_id():
    """A fixture to generate a fake custom model version ID."""

    return str(ObjectId())


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
        dr_client,
        regression_model_info,
        custom_models_url,
        regression_model_response,
    ):
        """A case to test a successful custom model creation."""

        responses.add(responses.POST, custom_models_url, json=regression_model_response, status=201)
        custom_model = dr_client.create_custom_model(regression_model_info)
        assert custom_model is not None

    @responses.activate
    def test_create_custom_model_failure(self, dr_client, regression_model_info, custom_models_url):
        """A case to test a failure in custom model creation."""

        status_code = 422
        responses.add(responses.POST, custom_models_url, json={}, status=status_code)
        with pytest.raises(DataRobotClientError) as ex:
            dr_client.create_custom_model(regression_model_info)
        assert ex.value.code == status_code

    @responses.activate
    def test_delete_custom_model_success(
        self,
        dr_client,
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
        dr_client.delete_custom_model_by_user_provided_id(expected_model["userProvidedId"])

    @responses.activate
    def test_delete_custom_model_failure(
        self,
        dr_client,
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
        dr_client,
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
                "customModelId": custom_model_id,
                "created": "2022-1-06 13:36:00",
                "items": [
                    {
                        "id": "629741dc5621557833bd5aa1",
                        "fileName": "custom.py",
                        "filePath": "custom.py",
                        "fileSource": "s3",
                    },
                    {
                        "id": "629741dc5621557833bd5aa2",
                        "fileName": "util/helper.py",
                        "filePath": "util/helper.py",
                        "fileSource": "s3",
                    },
                ],
                "isFrozen": False,
                "versionMajor": 1,
                "versionMinor": 2,
                "label": "1.2",
                "baseEnvironmentId": "629741dc5621557833bd5aa3",
                "gitModelVersion": {
                    "refName": ref_name,
                    "commitUrl": commit_url,
                    "mainBranchCommitSha": main_branch_commit_sha,
                    "pullRequestCommitSha": pull_request_commit_sha,
                },
                "dependencies": [],
            }

        return _inner

    @pytest.fixture
    def regression_model_version_response(self, regression_model_version_response_factory):
        """A fixture to return a specific Regression model version."""

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
        dr_client,
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
        dr_client,
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
        dr_client,
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
        total_versions_response = dr_client.fetch_custom_model_versions(custom_model_id)
        assert len(total_versions_response) == total_num_model_versions

        total_expected_versions = []
        for versions_per_page in expected_versions_in_all_pages.values():
            total_expected_versions.extend(versions_per_page)

        for fetched_version in total_versions_response:
            assert fetched_version in total_expected_versions

    class TestCustomModelsTestingRoute:
        """Contains unit-tests to test custom model testing routes."""

        @pytest.fixture
        def default_checks_config(self):
            """
            A fixture that returns the default configuration for custom model testing checks,
            when checks are disabled or do not exist in the model's YAML definition.
            """

            return {
                "errorCheck": "fail",
                "longRunningService": "fail",
                "nullValueImputation": "skip",
                "performanceCheck": "skip",
                "predictionVerificationCheck": "skip",
                "sideEffects": "skip",
                "stabilityCheck": "skip",
            }

        @pytest.mark.parametrize("loaded_checks", [None, {}], ids=["none", "empty_dict"])
        def test_minimal_custom_model_testing_configuration(
            self, loaded_checks, default_checks_config
        ):
            """A case to test minimal configuration for a custom model testing."""

            configuration = DrClient._build_tests_configuration(loaded_checks)
            assert configuration == default_checks_config

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
            self, mock_full_custom_model_checks, default_checks_config
        ):
            """
            A case to test a full custom model testing configuration, when all the checks are
            disabled.
            """

            assert mock_full_custom_model_checks.keys() == DrApiAttrs.DR_TEST_CHECK_MAP.keys()
            for _, info in mock_full_custom_model_checks.items():
                info[ModelSchema.CHECK_ENABLED_KEY] = False
            configuration = DrClient._build_tests_configuration(mock_full_custom_model_checks)
            assert configuration == default_checks_config

        def test_full_custom_model_testing_parameters(self, mock_full_custom_model_checks):
            """A case to test a full number of parameters in custom model testing."""

            assert mock_full_custom_model_checks.keys() == DrApiAttrs.DR_TEST_CHECK_MAP.keys()
            parameters = DrClient._build_tests_parameters(mock_full_custom_model_checks)
            dr_stability_check_key = DrApiAttrs.to_dr_test_check(ModelSchema.STABILITY_KEY)

            assert (
                parameters[dr_stability_check_key]["passingRate"]
                == mock_full_custom_model_checks[ModelSchema.STABILITY_KEY][
                    ModelSchema.PASSING_RATE_KEY
                ]
                / 100
            )

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

        @pytest.fixture
        def patch_dr_client_logging_level_for_debugging(self):
            """A fixture to path dr_client logger level to debug."""

            origin_level = dr_client_logger.getEffectiveLevel()
            dr_client_logger.setLevel(logging.DEBUG)
            yield
            dr_client_logger.setLevel(origin_level)

        @pytest.fixture
        def mock_custom_model_testing_input_args(self):
            """A fixture to mock input arguments for the custom model testing method."""

            return Mock(
                model_id="123",
                model_version_id="456",
                model_info=Mock(user_provided_id="789", model_path="/a/b/c"),
            )

        @contextlib.contextmanager
        def _mock_custom_model_version_response_factory(self, response_data):
            with patch.object(DrClient, "_post_custom_model_test_request"), patch.object(
                DrClient, "_wait_for_async_resolution"
            ), patch.object(
                HttpRequester, "get", return_value=Mock(json=Mock(return_value=response_data))
            ):
                yield

        @pytest.mark.usefixtures("patch_dr_client_logging_level_for_debugging")
        def test_custom_model_testing_success_response(
            self, dr_client, mock_custom_model_testing_input_args, caplog
        ):
            """A case to test a custom model testing successful response."""

            input_args = mock_custom_model_testing_input_args
            response_data = {"overallStatus": "succeeded"}
            with self._mock_custom_model_version_response_factory(response_data), patch.object(
                DrClient, "_analyse_custom_model_testing_checks_response"
            ) as analyse_method:
                dr_client.run_custom_model_version_testing(
                    model_id=input_args.model_id,
                    model_version_id=input_args.model_version_id,
                    model_info=input_args.model_info,
                )
                analyse_method.assert_not_called()
                assert (
                    "Custom model testing pass with success. "
                    f"User provided ID: {input_args.model_info.user_provided_id}."
                    in caplog.messages[0]
                )

        @pytest.mark.usefixtures("patch_dr_client_logging_level_for_debugging")
        def test_custom_model_testing_warning(
            self, dr_client, mock_custom_model_testing_input_args, caplog
        ):
            """A case to test a custom model testing response with warning."""

            input_args = mock_custom_model_testing_input_args
            response_data = {
                "overallStatus": "warning",
                "testingStatus": {
                    "longRunningService": {"status": "success"},
                    "errorCheck": {"status": "success"},
                    "performanceCheck": {"status": "warning"},
                },
            }
            with self._mock_custom_model_version_response_factory(response_data):
                dr_client.run_custom_model_version_testing(
                    model_id=input_args.model_id,
                    model_version_id=input_args.model_version_id,
                    model_info=input_args.model_info,
                )
                assert (
                    "Custom model version overall testing status, "
                    f"model_path: {input_args.model_info.model_path}, "
                    f"model_version_id: {input_args.model_version_id}, status: warning."
                    in caplog.messages[0]
                )
                for index, (key, value) in enumerate(response_data["testingStatus"].items()):
                    assert (
                        f"Check status, check '{key}', status: {value['status']}."
                        in caplog.messages[index + 1]
                    )
                assert (
                    "Custom model testing pass with success. "
                    f"User provided ID: {input_args.model_info.user_provided_id}."
                    in caplog.messages[-1]
                )

        @pytest.mark.usefixtures("patch_dr_client_logging_level_for_debugging")
        def test_custom_model_testing_failed(
            self, dr_client, mock_custom_model_testing_input_args, caplog
        ):
            """A case to test a custom model testing failure response."""

            input_args = mock_custom_model_testing_input_args
            expected_error_message = "Failed in error-check."
            response_data = {
                "overallStatus": "failed",
                "testingStatus": {
                    "longRunningService": {"status": "success"},
                    "errorCheck": {"status": "failed", "message": expected_error_message},
                    "performanceCheck": {"status": "warning"},
                },
            }
            with self._mock_custom_model_version_response_factory(response_data):
                with pytest.raises(DataRobotClientError) as ex:
                    dr_client.run_custom_model_version_testing(
                        model_id=input_args.model_id,
                        model_version_id=input_args.model_version_id,
                        model_info=input_args.model_info,
                    )
                assert expected_error_message in str(ex)
                assert (
                    "Custom model version overall testing status, "
                    f"model_path: {input_args.model_info.model_path}, "
                    f"model_version_id: {input_args.model_version_id}, status: failed."
                    in caplog.messages[0]
                )


class TestCustomModelVersionDependencies:
    """
    Contains unit-test to test the DataRobot custom model version dependencies and depdency
    environment.
    """

    @pytest.fixture
    def regression_model_version_factory(self, custom_model_id, custom_model_version_id):
        """A factory fixture to create a Regression model version w/o dependencies."""

        def _inner(with_dependencies):
            cm_version_response = {
                "id": custom_model_version_id,
                "customModelId": custom_model_id,
                "created": "2022-1-06 13:36:00",
                "items": [
                    {
                        "id": "629741dc5621557833bd5aa1",
                        "fileName": "custom.py",
                        "filePath": "custom.py",
                        "fileSource": "s3",
                    },
                ],
                "isFrozen": False,
                "versionMajor": 1,
                "versionMinor": 2,
                "label": "1.2",
                "baseEnvironmentId": "629741dc5621557833bd5aa3",
            }

            if with_dependencies:
                cm_version_response["items"].append(
                    {
                        "id": "629741dc5621557833bd5aa2",
                        "fileName": "requirements.txt",
                        "filePath": "requirements.txt",
                        "fileSource": "s3",
                    },
                )
                cm_version_response["dependencies"] = [
                    {
                        "packageName": "requests",
                        "line": "requests == 2.28.1",
                        "lineNumber": 2,
                        "constraints": [{"version": "2.28.1", "constraintType": "=="}],
                    }
                ]
            else:
                cm_version_response["dependencies"] = []

            return cm_version_response

        return _inner

    @pytest.fixture
    def dependency_environment_build_response_factory(self):
        """A fixture factory to create a dependency environment build response."""

        def _inner(build_status):
            """
            Returns build status response.

            Parameters
            ----------
            build_status : str
                The build status: one of ["submitted", "processing", "success", "failed"].

            Returns
            -------
            dict :
                Build status response.

            """
            assert build_status in ["submitted", "processing", "success", "failed"]
            return {
                "buildStatus": build_status,
                "buildStart": "2022-11-08T13:47:26.577146Z",
                "buildEnd": (
                    None
                    if build_status in ["submitted", "processing"]
                    else "2022-11-08T18:47:26.577146Z"
                ),
                "buildLogLocation": (
                    None
                    if build_status in ["submitted", "processing"]
                    else "https://dr/api/v2/customModels/6357/versions/636/dependencyBuildLog/"
                ),
            }

        return _inner

    @pytest.fixture
    def dependency_build_url(self, webserver, custom_model_id, custom_model_version_id):
        """A fixture to return the dependency environment build URL."""

        url_path = DrClient.CUSTOM_MODELS_VERSION_DEPENDENCY_BUILD_ROUTE.format(
            model_id=custom_model_id, model_ver_id=custom_model_version_id
        )
        return f"{webserver}/api/v2/{url_path}"

    def test_dependency_environment_build_is_not_required(
        self, dr_client, regression_model_version_factory
    ):
        """Test the case in which a dependency environment built is not required."""

        cm_version = regression_model_version_factory(with_dependencies=False)
        with patch.object(
            DrClient, "_dependency_environment_already_built_or_in_progress"
        ) as mock_method:
            dr_client.build_dependency_environment_if_required(cm_version)
            mock_method.assert_not_called()

    @pytest.mark.parametrize("build_status", ["submitted", "processing", "success", "failed"])
    @responses.activate
    def test_dependency_environment_was_built_or_in_progress(
        self,
        dr_client,
        dependency_build_url,
        regression_model_version_factory,
        dependency_environment_build_response_factory,
        build_status,
    ):
        """
        Test the case in which a dependency environment build was already completed or in progress.
        """

        build_status_response = dependency_environment_build_response_factory(build_status)
        response = responses.get(dependency_build_url, json=build_status_response, status=200)
        cm_version = regression_model_version_factory(with_dependencies=True)

        with patch.object(DrClient, "_monitor_dependency_environment_building") as mock_method:
            dr_client.build_dependency_environment_if_required(cm_version)
            assert response.call_count == 1
            mock_method.assert_not_called()

    @responses.activate
    def test_dependency_environment_build_started_and_succeeded(
        self,
        dr_client,
        custom_model_version_id,
        dependency_build_url,
        regression_model_version_factory,
        dependency_environment_build_response_factory,
    ):
        """
        Test the case in which a dependency environment build was submitted and eventually
        succeeded.
        """

        response_objs = self._setup_ordered_responses(
            custom_model_version_id,
            dependency_build_url,
            dependency_environment_build_response_factory,
            is_success=True,
        )
        cm_version = regression_model_version_factory(with_dependencies=True)
        with patch("time.sleep", return_value=None):
            dr_client.build_dependency_environment_if_required(cm_version)
        for response_obj in response_objs:
            assert response_obj.call_count == 1

    @staticmethod
    def _setup_ordered_responses(
        custom_model_version_id,
        dependency_build_url,
        dependency_environment_build_response_factory,
        is_success,
    ):
        build_status = "success" if is_success else "failed"
        ordered_responses = [
            (
                responses.GET,
                {
                    "message": f"Custom model version {custom_model_version_id} build has not "
                    "been started"
                },
                422,
            ),
            (
                responses.POST,
                {
                    "buildStatus": "submitted",
                    "buildStart": "2022-11-08T13:47:26.577146Z",
                    "buildEnd": None,
                    "buildLogLocation": None,
                },
                202,
            ),
            (responses.GET, dependency_environment_build_response_factory("submitted"), 200),
            (responses.GET, dependency_environment_build_response_factory("processing"), 200),
            (responses.GET, dependency_environment_build_response_factory("processing"), 200),
            (responses.GET, dependency_environment_build_response_factory(build_status), 200),
        ]
        response_objs = []
        for method, response_json, status_code in ordered_responses:
            response_obj = responses.add(  # pylint: disable=assignment-from-none
                method, dependency_build_url, json=response_json, status=status_code
            )
            response_objs.append(response_obj)
        return response_objs

    @responses.activate
    def test_dependency_environment_build_started_and_failed(
        self,
        webserver,
        dr_client,
        custom_model_id,
        custom_model_version_id,
        dependency_build_url,
        regression_model_version_factory,
        dependency_environment_build_response_factory,
    ):
        """
        Test the case in which a dependency environment build was submitted but eventually failed.
        """

        response_objs = self._setup_ordered_responses(
            custom_model_version_id,
            dependency_build_url,
            dependency_environment_build_response_factory,
            is_success=False,
        )
        url_sub_path = DrClient.CUSTOM_MODELS_VERSION_DEPENDENCY_BUILD_LOG_ROUTE.format(
            model_id=custom_model_id, model_ver_id=custom_model_version_id
        )
        dependency_build_log_url = f"{webserver}/api/v2/{url_sub_path}"
        build_error_message = """
        Step 1/4 : FROM registry.cm-staging.int.datarobot.com/custom-models/base-image:6356
         ---> 8b6667fecb60
        Step 2/4 : RUN pip install   "requests==2.30.0"
         ---> Running in c4b112ff358c
        ERROR: Could not find a version that satisfies the requirement requests==2.30.0
        ERROR: No matching distribution found for requests==2.30.0
        [notice] A new release of pip available: 22.3 -> 22.3.1
        [notice] To update, run: python3 -m pip install --upgrade pip
        Removing intermediate container c4b112ff358c
        """
        response_obj = responses.get(dependency_build_log_url, body=build_error_message, status=200)

        cm_version = regression_model_version_factory(with_dependencies=True)
        with patch("time.sleep", return_value=None), pytest.raises(DataRobotClientError) as ex:
            dr_client.build_dependency_environment_if_required(cm_version)
        assert build_error_message in str(ex.value)
        assert response_obj.call_count == 1
        for response_obj in response_objs:
            assert response_obj.call_count == 1


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
        dr_client,
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
        total_deployments_response = dr_client.fetch_deployments()
        assert len(total_deployments_response) == total_num_deployments

        total_expected_deployments = []
        for deployments_per_page in expected_deployments_in_all_pages.values():
            total_expected_deployments.extend(deployments_per_page)

        for fetched_deployment in total_deployments_response:
            assert fetched_deployment in total_expected_deployments


class TestDeploymentPayloadConstruction:
    """A class to test the deployment's payload construction method."""

    def test_deployment_update_payload_construction_full(self, dr_client):
        """A case to test a deployment update payload construction, with all config changes."""

        attr_map = {
            DeploymentSchema.LABEL_KEY: {"desired": "New label", "actual": "Origin label"},
            DeploymentSchema.DESCRIPTION_KEY: {
                "desired": "New description",
                "actual": "Origin description",
            },
            DeploymentSchema.IMPORTANCE_KEY: {
                "desired": DeploymentSchema.IMPORTANCE_MODERATE_VALUE,
                "actual": DeploymentSchema.IMPORTANCE_LOW_VALUE,
            },
        }
        deployment = {attr: body["actual"] for attr, body in attr_map.items()}
        payload = dr_client._construct_deployment_update_payload(
            deployment, self._deployment_info(attr_map)
        )
        assert payload == {attr: body["desired"] for attr, body in attr_map.items()}

    @staticmethod
    def _deployment_info(attr_map):
        deployment_metadata = {
            DeploymentSchema.SETTINGS_SECTION_KEY: {
                attr: body["desired"] for attr, body in attr_map.items()
            }
        }
        return DeploymentInfo("dummy.yaml", deployment_metadata)

    def test_deployment_update_payload_construction_no_changes(self, dr_client):
        """A case to test a deployment update payload construction when no config changes."""

        label_value = "Origin label"
        description_value = "Origin description"
        importance_value = DeploymentSchema.IMPORTANCE_LOW_VALUE
        attr_map = {
            DeploymentSchema.LABEL_KEY: {"desired": label_value, "actual": label_value},
            DeploymentSchema.DESCRIPTION_KEY: {
                "desired": description_value,
                "actual": description_value,
            },
            DeploymentSchema.IMPORTANCE_KEY: {
                "desired": importance_value,
                "actual": importance_value,
            },
        }
        deployment = {attr: body["actual"] for attr, body in attr_map.items()}
        payload = dr_client._construct_deployment_update_payload(
            deployment, self._deployment_info(attr_map)
        )
        assert not payload

    @pytest.mark.parametrize("empty_label", [None, ""], ids=["none", "empty"])
    def test_deployment_update_payload_construction_empty_label(self, dr_client, empty_label):
        """A case to test a deployment update payload construction with empty label."""

        attr_map = {
            DeploymentSchema.LABEL_KEY: {"desired": empty_label, "actual": "Origin label"},
            DeploymentSchema.DESCRIPTION_KEY: {
                "desired": "Origin description",
                "actual": "Origin description",
            },
            DeploymentSchema.IMPORTANCE_KEY: {
                "desired": DeploymentSchema.IMPORTANCE_LOW_VALUE,
                "actual": DeploymentSchema.IMPORTANCE_LOW_VALUE,
            },
        }
        deployment = {attr: body["actual"] for attr, body in attr_map.items()}
        payload = dr_client._construct_deployment_update_payload(
            deployment, self._deployment_info(attr_map)
        )
        assert not payload

    @pytest.mark.parametrize("description", [None, "", "Some new description"])
    def test_deployment_update_payload_construction_description(self, dr_client, description):
        """
        A case to test a deployment update payload construction with different description
        attribute values.
        """

        attr_map = {
            DeploymentSchema.LABEL_KEY: {"desired": None, "actual": "Origin label"},
            DeploymentSchema.DESCRIPTION_KEY: {
                "desired": description,
                "actual": "Origin description",
            },
            DeploymentSchema.IMPORTANCE_KEY: {
                "desired": None,
                "actual": DeploymentSchema.IMPORTANCE_LOW_VALUE,
            },
        }
        deployment = {attr: body["actual"] for attr, body in attr_map.items()}
        payload = dr_client._construct_deployment_update_payload(
            deployment, self._deployment_info(attr_map)
        )
        assert payload[DeploymentSchema.DESCRIPTION_KEY] == description

    @pytest.mark.parametrize("empty_importance", [None, ""], ids=["none", "empty"])
    def test_deployment_update_payload_construction_empty_importance(
        self, dr_client, empty_importance
    ):
        """
        A case to test a deployment update payload construction with empty importance attribute
        value.
        """

        attr_map = {
            DeploymentSchema.LABEL_KEY: {"desired": None, "actual": "Origin label"},
            DeploymentSchema.DESCRIPTION_KEY: {
                "desired": "Origin description",
                "actual": "Origin description",
            },
            DeploymentSchema.IMPORTANCE_KEY: {
                "desired": empty_importance,
                "actual": DeploymentSchema.IMPORTANCE_LOW_VALUE,
            },
        }
        deployment = {attr: body["actual"] for attr, body in attr_map.items()}
        payload = dr_client._construct_deployment_update_payload(
            deployment, self._deployment_info(attr_map)
        )
        assert not payload


class TestDeploymentSettingsPayloadConstruction:
    """A class to test the deployment's settings payload contruction method."""

    CHALLENGER_ENABLED_SUB_PAYLOAD = {
        "challengerModels": {"enabled": True},
        "predictionsDataCollection": {"enabled": True},
    }

    def test_empty_settings__none_actual_settings(self, dr_client):
        """A case to test an empty settings with no actual settings."""

        deployment_info_no_settings = DeploymentInfo("dummy.yaml", {})
        payload = dr_client._construct_deployment_settings_payload(deployment_info_no_settings)
        assert payload == self.CHALLENGER_ENABLED_SUB_PAYLOAD

    @pytest.mark.parametrize("challenger_enabled", [None, True, False])
    @pytest.mark.parametrize("prediction_data_collection_enabled", [None, True, False])
    def test_challenger_and_pred_data_collection_config__none_actual_settings(
        self, dr_client, challenger_enabled, prediction_data_collection_enabled
    ):
        """
        A case to test the challenger and prediction data collection configuration, when
        no actual settings.
        """

        deployment_info = DeploymentInfo(
            "dummy.yaml",
            {
                DeploymentSchema.SETTINGS_SECTION_KEY: {
                    DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY: challenger_enabled,
                    DeploymentSchema.ENABLE_PREDICTIONS_COLLECTION_KEY: (
                        prediction_data_collection_enabled
                    ),
                },
            },
        )
        payload = dr_client._construct_deployment_settings_payload(deployment_info)
        if challenger_enabled or challenger_enabled is None:
            assert payload == self.CHALLENGER_ENABLED_SUB_PAYLOAD
        else:
            if not prediction_data_collection_enabled:
                assert not payload
            else:
                assert payload == {"predictionsDataCollection": {"enabled": True}}

    @pytest.mark.parametrize("desired_challenger_enabled", [None, True, False])
    @pytest.mark.parametrize("actual_challenger_enabled", [None, True, False])
    @pytest.mark.parametrize("desired_pred_data_collection_enabled", [None, True, False])
    @pytest.mark.parametrize("actual_pred_data_collection_enabled", [None, True, False])
    def test_challenger_and_pred_data_collection_config_with_actual_settings(
        self,
        dr_client,
        desired_challenger_enabled,
        actual_challenger_enabled,
        desired_pred_data_collection_enabled,
        actual_pred_data_collection_enabled,
    ):
        """
        A case to test the challenger and prediction data collection configuration when actual
        settings are available.
        """

        deployment_info_no_settings = DeploymentInfo(
            "dummy.yaml",
            {
                DeploymentSchema.SETTINGS_SECTION_KEY: {
                    DeploymentSchema.ENABLE_CHALLENGER_MODELS_KEY: desired_challenger_enabled,
                    DeploymentSchema.ENABLE_PREDICTIONS_COLLECTION_KEY: (
                        desired_pred_data_collection_enabled
                    ),
                },
            },
        )
        actual_settings = self._actual_settings(
            actual_challenger_enabled, actual_pred_data_collection_enabled
        )
        payload = dr_client._construct_deployment_settings_payload(
            deployment_info_no_settings, actual_settings
        )
        effective_desired_challenger = (
            desired_challenger_enabled or desired_challenger_enabled is None
        )
        effective_actual_challenger = actual_settings and actual_settings.get(
            "challengerModels", {}
        ).get("enabled")
        if effective_actual_challenger:
            if effective_desired_challenger:
                if actual_settings.get("predictionsDataCollection", {}).get("enabled"):
                    # Actual and desired are enabled for both challenger and prediction data
                    # collection
                    assert not payload
                else:
                    # Only actual and desired challenger are enabled
                    assert payload == {"predictionsDataCollection": {"enabled": True}}
            else:
                expected_payload = {"challengerModels": {"enabled": False}}
                if desired_pred_data_collection_enabled != actual_settings.get(
                    "predictionsDataCollection", {}
                ).get("enabled"):
                    expected_payload["predictionsDataCollection"] = {
                        "enabled": bool(desired_pred_data_collection_enabled)
                    }
                assert payload == expected_payload
        else:
            if effective_desired_challenger:
                if actual_pred_data_collection_enabled:
                    assert payload == {"challengerModels": {"enabled": True}}
                else:
                    assert payload == self.CHALLENGER_ENABLED_SUB_PAYLOAD
            else:
                if bool(desired_pred_data_collection_enabled) != bool(
                    actual_pred_data_collection_enabled
                ):
                    assert payload == {
                        "predictionsDataCollection": {
                            "enabled": bool(desired_pred_data_collection_enabled)
                        }
                    }
                else:
                    assert not payload

    @staticmethod
    def _actual_settings(challenger_enabled, pred_data_collection_enabled):
        if challenger_enabled is None and pred_data_collection_enabled is None:
            return None

        actual_settings = {}
        if challenger_enabled is not None:
            actual_settings = {"challengerModels": {"enabled": challenger_enabled}}

        if challenger_enabled:
            actual_settings["predictionsDataCollection"] = {"enabled": True}
        else:
            if pred_data_collection_enabled is not None:
                actual_settings["predictionsDataCollection"] = {
                    "enabled": pred_data_collection_enabled
                }
        return actual_settings
