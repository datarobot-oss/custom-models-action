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
from abc import ABC
from abc import abstractmethod

import mock
import pytest
import responses
import schema
from bson import ObjectId
from mock import Mock
from mock import patch
from responses import matchers

from common.exceptions import DataRobotClientError
from common.http_requester import HttpRequester
from common.namepsace import Namespace
from deployment_info import DeploymentInfo
from dr_api_attrs import DrApiCustomModelChecks
from dr_api_attrs import DrApiModelSettings
from dr_client import DrClient
from dr_client import logger as dr_client_logger
from model_file_path import ModelFilePath
from model_info import ModelInfo
from schema_validator import DeploymentSchema
from schema_validator import ModelSchema
from tests.unit.conftest import set_namespace


@pytest.fixture(name="webserver")
def fixture_webserver():
    """A fixture to return a fake DataRobot webserver."""

    return "http://www.datarobot.dummy-app"


@pytest.fixture(name="api_token")
def fixture_api_token():
    """A fixture to return a fake API token."""

    return "123abc"


@pytest.fixture(name="dr_client")
def dr_client_fixture(webserver, api_token):
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
        ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION,
        ModelSchema.SETTINGS_SECTION_KEY: {
            ModelSchema.NAME_KEY: "minimal-regression-model",
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
        metadata=ModelSchema.validate_and_transform_single(metadata),
    )


@pytest.fixture(name="regression_model_info")
def fixture_regression_model_info():
    """A fixture to create a local ModelInfo with information of a regression model."""

    metadata = {
        ModelSchema.MODEL_ID_KEY: "abc123",
        ModelSchema.TARGET_TYPE_KEY: ModelSchema.TARGET_TYPE_REGRESSION,
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
            ModelSchema.EGRESS_NETWORK_POLICY_KEY: ModelSchema.EGRESS_NETWORK_POLICY_PUBLIC,
        },
    }
    return ModelInfo(
        yaml_filepath="/dummy/yaml/filepath",
        model_path="/dummy/model/path",
        metadata=metadata,
    )


def mock_paginated_responses(
    total_num_entities, num_entities_in_page, url_factory, entity_response_factory_fn, match=None
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
            match=match or [],
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


def mock_single_page_response(url, entities, match=None):
    """Mock single page paginated response."""

    def url_factory(_):
        return url

    entities_iter = iter(entities)

    def entity_response_factory_fn(_):
        return next(entities_iter)

    return mock_paginated_responses(
        len(entities), max(len(entities), 1), url_factory, entity_response_factory_fn, match
    )


class TestPaginator:
    """
    A class to test the DrClient when it fetched entities from DataRobot from URLs that support
    pagination.
    """

    @pytest.fixture
    def mock_response_with_two_pages(self):
        """A fixture to mock a response for get method with two sequenced paginated pages."""

        total_entities_num = 5
        page_one_num_entities = 3
        dummy_responses = [
            Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "totalCount": total_entities_num,
                        "count": 3,
                        "next": "https://next-page",
                        "data": list(range(page_one_num_entities)),
                    }
                ),
            ),
            Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "totalCount": total_entities_num,
                        "count": total_entities_num - 3,
                        "next": None,
                        "data": list(range(page_one_num_entities, total_entities_num)),
                    }
                ),
            ),
        ]
        with patch.object(HttpRequester, "get", side_effect=dummy_responses) as mock_get:
            yield mock_get, total_entities_num

    def test_paginator_success_without_arguments(self, mock_response_with_two_pages):
        """A case to test fetching without arguments of paginated pages."""

        mock_get, total_entities_num = mock_response_with_two_pages
        dr_client = DrClient("https://dummy", "123abc")
        total_entities_in_response = dr_client._paginated_fetch("/deployments")
        assert total_entities_num == len(total_entities_in_response)
        mock_get.assert_has_calls(
            [mock.call("/deployments", False), mock.call("https://next-page", True)]
        )

    def test_paginator_success_with_json(self, mock_response_with_two_pages):
        """A case to test fetching with 'json' argument of paginated pages."""

        mock_get, total_entities_num = mock_response_with_two_pages
        dr_client = DrClient("https://dummy", "123abc")
        total_entities_in_response = dr_client._paginated_fetch("/deployments", json={"x": 3})
        assert total_entities_num == len(total_entities_in_response)
        mock_get.assert_has_calls(
            [
                mock.call("/deployments", False, json={"x": 3}),
                mock.call("https://next-page", True, json={"x": 3}),
            ]
        )

    def test_paginator_success_with_params(self, mock_response_with_two_pages):
        """A case to test fetching with 'params' argument of paginated pages."""

        mock_get, total_entities_num = mock_response_with_two_pages
        dr_client = DrClient("https://dummy", "123abc")
        total_entities_in_response = dr_client._paginated_fetch("/deployments", params={"x": 3})
        assert total_entities_num == len(total_entities_in_response)
        mock_get.assert_called_with("https://next-page", True)
        mock_get.assert_has_calls(
            [
                mock.call("/deployments", False, params={"x": 3}),
                mock.call("https://next-page", True),
            ]
        )


# pylint: disable=too-few-public-methods
class SharedRouteTests(ABC):
    """A base class for the model and deployment routes tests."""

    @abstractmethod
    def _fetch_entities(self, dr_client):
        raise NotImplementedError("Only derived classes should implement this.")

    def _test_fetch_entities_success(
        self, dr_client, total_num_entities, num_entities_in_page, url_factory, response_factory
    ):
        expected_entities_in_all_pages = mock_paginated_responses(
            total_num_entities, num_entities_in_page, url_factory, response_factory
        )
        total_entities_response = self._fetch_entities(dr_client)

        total_expected_entities = []
        for entities_per_page in expected_entities_in_all_pages.values():
            total_expected_entities.extend(entities_per_page)

        for fetched_entity in total_entities_response:
            assert fetched_entity in total_expected_entities

    @staticmethod
    def _test_fetch_entities_with_multiple_namespace_success(
        url_factory_fn, response_factory_fn, fetch_fn
    ):
        num_entities_in_each_namespace = 3
        total_entities = []
        for namespace in [None, "dev-1", "dev-2"]:
            with set_namespace(namespace):
                for index in range(num_entities_in_each_namespace):
                    total_entities.append(response_factory_fn(index))
                responses.add(
                    responses.GET,
                    url_factory_fn(0),
                    json={
                        "totalCount": len(total_entities),
                        "count": len(total_entities),
                        "next": None,
                        "data": total_entities,
                    },
                    status=200,
                )
                total_entities_response = fetch_fn()
                if namespace is None:
                    assert len(total_entities_response) == len(total_entities)
                else:
                    assert len(total_entities_response) == num_entities_in_each_namespace
                assert all(
                    Namespace.is_in_namespace(e["userProvidedId"]) for e in total_entities_response
                )


class TestCustomModelRoutes(SharedRouteTests):
    """Contains cases to test DataRobot custom models routes."""

    def _fetch_entities(self, dr_client):
        return dr_client.fetch_custom_models()

    @pytest.fixture
    def regression_model_response_factory(self):
        """A factory fixture to generate a regression custom model response."""

        def _inner(model_id):
            return {
                "id": model_id,
                "userProvidedId": Namespace.namespaced(f"user-provided-id-{model_id}"),
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

    @pytest.fixture
    def git_model_version(self):
        """A fixture to return a dummy git model version."""

        pull_request_commit_sha = "1" * 40
        return Mock(
            ref_name="feature-branch",
            commit_url=f"https://github.com/user/project/{pull_request_commit_sha}",
            main_branch_commit_sha="2" * 40,
            pull_request_commit_sha=pull_request_commit_sha,
        )

    def test_full_payload_setup_for_custom_model_creation(
        self, regression_model_info, git_model_version
    ):
        """A case to test full payload setup to create a custom model."""

        payload = DrClient._setup_payload_for_custom_model_creation(
            regression_model_info, git_model_version
        )
        self._validate_mandatory_attributes_for_regression_model(payload, optional_exist=True)

    @staticmethod
    def _validate_mandatory_attributes_for_regression_model(payload, optional_exist):
        assert "name" in payload
        assert "customModelType" in payload
        assert "targetType" in payload
        assert "targetName" in payload
        assert "isUnstructuredModelKind" in payload
        assert "userProvidedId" in payload
        assert "predictionThreshold" in payload
        assert ("description" in payload) == optional_exist
        assert ("language" in payload) == optional_exist

    def test_minimal_payload_setup_for_custom_model_creation(
        self, minimal_regression_model_info, git_model_version
    ):
        """A case to test a minimal payload setup to create a custom model."""

        payload = DrClient._setup_payload_for_custom_model_creation(
            minimal_regression_model_info, git_model_version
        )
        self._validate_mandatory_attributes_for_regression_model(payload, optional_exist=False)

    @responses.activate
    def test_create_custom_model_success(
        self,
        dr_client,
        regression_model_info,
        custom_models_url,
        regression_model_response,
        git_model_version,
    ):
        """A case to test a successful custom model creation."""

        responses.add(responses.POST, custom_models_url, json=regression_model_response, status=201)
        custom_model = dr_client.create_custom_model(regression_model_info, git_model_version)
        assert custom_model is not None

    @responses.activate
    def test_create_custom_model_failure(
        self, dr_client, regression_model_info, custom_models_url, git_model_version
    ):
        """A case to test a failure in custom model creation."""

        status_code = 422
        responses.add(responses.POST, custom_models_url, json={}, status=status_code)
        with pytest.raises(DataRobotClientError) as ex:
            dr_client.create_custom_model(regression_model_info, git_model_version)
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
        responses.add(responses.DELETE, delete_url, status=204)
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

        self._test_fetch_entities_success(
            dr_client,
            total_num_models,
            num_models_in_page,
            custom_models_url_factory,
            regression_model_response_factory,
        )

    @responses.activate
    def test_fetch_custom_models_with_multiple_namespaces_success(
        self, dr_client, custom_models_url_factory, regression_model_response_factory
    ):
        """A case to test a successful custom model retrieval with multiple namespaces."""

        self._test_fetch_entities_with_multiple_namespace_success(
            custom_models_url_factory,
            regression_model_response_factory,
            dr_client.fetch_custom_models,
        )


class TestCustomModelSettingsPayload:
    """Contains cases to test custom model settings payload construction."""

    @pytest.fixture
    def model_info(self):
        """A fixture to create a ModelInfo, which only includes the settings section."""

        metadata = {
            ModelSchema.SETTINGS_SECTION_KEY: {
                ModelSchema.NAME_KEY: "my-model",
                ModelSchema.DESCRIPTION_KEY: "Some description",
                ModelSchema.LANGUAGE_KEY: "Python",
                ModelSchema.TARGET_NAME_KEY: "Some Target",
                ModelSchema.PREDICTION_THRESHOLD_KEY: 0.5,
            }
        }
        return ModelInfo(yaml_filepath="/tmp/dummy", model_path="/tmp/model", metadata=metadata)

    @pytest.fixture
    def datarobot_custom_model(self):
        """
        A fixture to create a response custom model dict with the settings related attributes only.
        """

        return {
            DrApiModelSettings.to_dr_attr(ModelSchema.NAME_KEY): "my-model",
            DrApiModelSettings.to_dr_attr(ModelSchema.DESCRIPTION_KEY): "Some description",
            DrApiModelSettings.to_dr_attr(ModelSchema.LANGUAGE_KEY): "Python",
            DrApiModelSettings.to_dr_attr(ModelSchema.TARGET_NAME_KEY): "Some Target",
            DrApiModelSettings.to_dr_attr(ModelSchema.PREDICTION_THRESHOLD_KEY): 0.5,
        }

    def test_no_update(self, model_info, datarobot_custom_model):
        """Test no settings differences between local and remote model."""

        payload = DrClient.get_settings_patch_payload(model_info, datarobot_custom_model)
        assert not payload

    def test_missing_optional_attribute_which_results_in_no_update(
        self, model_info, datarobot_custom_model
    ):
        """
        Test missing local optional attributes, which result in no update to the model's settings.
        """

        model_info.metadata[ModelSchema.SETTINGS_SECTION_KEY].pop(ModelSchema.DESCRIPTION_KEY)

        payload = DrClient.get_settings_patch_payload(model_info, datarobot_custom_model)
        assert not payload

    def test_single_attribute_change(self, model_info, datarobot_custom_model):
        """
        Test a single settings attribute change, which result in an update of that specific
        attribute.
        """

        origin_name = model_info.get_settings_value(ModelSchema.NAME_KEY)
        new_name = f"{origin_name}-new"
        model_info.metadata[ModelSchema.SETTINGS_SECTION_KEY][ModelSchema.NAME_KEY] = new_name

        payload = DrClient.get_settings_patch_payload(model_info, datarobot_custom_model)
        assert payload[DrApiModelSettings.to_dr_attr(ModelSchema.NAME_KEY)] == new_name


class TestCustomModelTrainingHoldoutPayload:
    """
    Contains test cases for building an update payload for training and holdout data, for both
    structured and unstructured models.
    """

    @pytest.fixture
    def training_dataset_id(self):
        """A fixture to return a training dataset ID."""

        return "123"

    @pytest.fixture
    def partitioning_column(self):
        """A fixture to return a partitioning column, which is used in structured models."""

        return "partitioning-col"

    @pytest.fixture
    def holdout_dataset_id(self):
        """A fixture to return a holdout dataset ID, which is used in unstructured models."""

        return "456"

    @pytest.fixture
    def datarobot_structured_model(self, training_dataset_id, partitioning_column):
        """A fixture that returns the training/holdout portion of a datarobot structured model."""

        return {
            "trainingDatasetId": training_dataset_id,
            "trainingDataPartitionColumn": partitioning_column,
        }

    @pytest.fixture
    def datarobot_unstructured_model(self, training_dataset_id, holdout_dataset_id):
        """A fixture that returns the training/holdout portion of a datarobot unstructured model."""

        return {
            "externalMlopsStatsConfig": {
                "trainingDatasetId": training_dataset_id,
                "holdoutDatasetId": holdout_dataset_id,
            }
        }

    @pytest.fixture
    def model_info_factory(self, training_dataset_id, partitioning_column, holdout_dataset_id):
        """
        A factory to create a ModelInfo, which represents the local configuration for
        training/holdout data.
        """

        def _inner(is_unstructured):
            settings_section = {
                ModelSchema.TRAINING_DATASET_ID_KEY: training_dataset_id,
            }
            metadata = {ModelSchema.SETTINGS_SECTION_KEY: settings_section}
            if is_unstructured:
                metadata[
                    ModelSchema.TARGET_TYPE_KEY
                ] = ModelSchema.TARGET_TYPE_UNSTRUCTURED_REGRESSION
                settings_section[ModelSchema.HOLDOUT_DATASET_ID_KEY] = holdout_dataset_id
            else:
                metadata[ModelSchema.TARGET_TYPE_KEY] = ModelSchema.TARGET_TYPE_REGRESSION
                settings_section[ModelSchema.PARTITIONING_COLUMN_KEY] = partitioning_column

            return ModelInfo(yaml_filepath="/tmp/dummy", model_path="/tmp/model", metadata=metadata)

        return _inner

    def test_structured_model_local_and_remote_are_the_same(
        self, datarobot_structured_model, model_info_factory
    ):
        """A case to test no settings changes for structured model."""

        model_info = model_info_factory(is_unstructured=False)
        payload = DrClient.get_training_holdout_patch_payload_at_model_level(
            model_info, datarobot_structured_model
        )
        assert not payload

    def test_structured_model_single_change(self, datarobot_structured_model, model_info_factory):
        """
        A case to test a single settings attribute change for structured model, which results in
        an update of that specific attribute only.
        """

        model_info = model_info_factory(is_unstructured=False)
        response_mapping = DrApiModelSettings.STRUCTURED_TRAINING_HOLDOUT_RESPONSE_MAPPING
        for local_key, remote_response_key in response_mapping.items():
            origin_value = model_info.get_settings_value(local_key)
            revised_value = f"{origin_value}a"
            with self._temporarily_revise_structured_model(
                datarobot_structured_model, remote_response_key, revised_value
            ):
                payload = DrClient.get_training_holdout_patch_payload_at_model_level(
                    model_info, datarobot_structured_model
                )
                patch_key = DrApiModelSettings.STRUCTURED_TRAINING_HOLDOUT_PATCH_MAPPING[local_key]
                assert payload == {patch_key: origin_value}

    @contextlib.contextmanager
    def _temporarily_revise_structured_model(self, datarobot_model, remote_key, revised_value):
        origin_value = datarobot_model[remote_key]
        datarobot_model[remote_key] = revised_value
        yield
        datarobot_model[remote_key] = origin_value

    def test_structured_model_full_update(self, datarobot_structured_model, model_info_factory):
        """
        A case to test full settings changes for structured model, which results in an update of
        the all the training/holdout data attributes.
        """

        model_info = model_info_factory(is_unstructured=False)
        response_mapping = DrApiModelSettings.STRUCTURED_TRAINING_HOLDOUT_RESPONSE_MAPPING
        for local_key, response_key in response_mapping.items():
            datarobot_structured_model[response_key] = None

        payload = DrClient.get_training_holdout_patch_payload_at_model_level(
            model_info, datarobot_structured_model
        )

        patch_mapping = DrApiModelSettings.STRUCTURED_TRAINING_HOLDOUT_PATCH_MAPPING
        for local_key, patch_key in patch_mapping.items():
            assert payload[patch_key] == model_info.get_settings_value(local_key)

    def test_unstructured_model_local_and_remote_are_the_same(
        self, datarobot_unstructured_model, model_info_factory
    ):
        """A case to test no settings changes for unstructured model."""

        model_info = model_info_factory(is_unstructured=True)
        payload = DrClient.get_training_holdout_patch_payload_at_model_level(
            model_info, datarobot_unstructured_model
        )
        assert not payload

    def test_unstructured_model_single_change(
        self, datarobot_unstructured_model, model_info_factory
    ):
        """
        A case to test a single settings attribute change for unstructured model, which results in
        an update of that specific attribute only.
        """

        model_info = model_info_factory(is_unstructured=True)
        keys_mapping = DrApiModelSettings.UNSTRUCTURED_TRAINING_HOLDOUT_MAPPING
        for local_key, remote_key in keys_mapping.items():
            origin_value = model_info.get_settings_value(local_key)
            revised_value = f"{origin_value}a"
            with self._temporarily_revise_unstructured_model(
                datarobot_unstructured_model, remote_key, revised_value
            ):
                payload = DrClient.get_training_holdout_patch_payload_at_model_level(
                    model_info, datarobot_unstructured_model
                )
                assert payload == {"externalMlopsStatsConfig": {remote_key: origin_value}}

    @contextlib.contextmanager
    def _temporarily_revise_unstructured_model(self, datarobot_model, remote_key, revised_value):
        origin_value = datarobot_model["externalMlopsStatsConfig"][remote_key]
        datarobot_model["externalMlopsStatsConfig"][remote_key] = revised_value
        yield
        datarobot_model["externalMlopsStatsConfig"][remote_key] = origin_value

    def test_unstructured_model_full_update(self, datarobot_unstructured_model, model_info_factory):
        """
        A case to test full settings changes for unstructured model, which results in an update of
        the all the training/holdout data attributes.
        """

        model_info = model_info_factory(is_unstructured=True)
        keys_mapping = DrApiModelSettings.UNSTRUCTURED_TRAINING_HOLDOUT_MAPPING
        ext_mlops_stats_config = datarobot_unstructured_model["externalMlopsStatsConfig"]
        for local_key, remote_key in keys_mapping.items():
            ext_mlops_stats_config[remote_key] = None
        payload = DrClient.get_training_holdout_patch_payload_at_model_level(
            model_info, datarobot_unstructured_model
        )
        ext_mlops_stats_config_payload = payload["externalMlopsStatsConfig"]
        for local_key, remote_key in keys_mapping.items():
            local_value = model_info.get_settings_value(local_key)
            assert ext_mlops_stats_config_payload[remote_key] == local_value


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
        """A fixture to return a dummy Git ref name."""

        return "feature-branch"

    @pytest.fixture
    def git_model_version(
        self, ref_name, commit_url, main_branch_commit_sha, pull_request_commit_sha
    ):
        """
        A fixture to return a mocked Git model version instance. Note that this fixture overrides
        another fixture with the same name at a higher level, which is in conftest.py
        """

        return Mock(
            ref_name=ref_name,
            commit_url=commit_url,
            main_branch_commit_sha=main_branch_commit_sha,
            pull_request_commit_sha=pull_request_commit_sha,
        )

    @pytest.fixture
    def commit_url(self, pull_request_commit_sha):
        """A fixture to return a dummy GitHub commit web URL."""

        return f"https://github.com/user/project/{pull_request_commit_sha}"

    @pytest.fixture
    def regression_model_version_response_factory(self, custom_model_id, git_model_version):
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
                "gitModelVersion": {  # pylint: disable=duplicate-code
                    "refName": git_model_version.ref_name,
                    "commitUrl": git_model_version.commit_url,
                    "mainBranchCommitSha": git_model_version.main_branch_commit_sha,
                    "pullRequestCommitSha": git_model_version.pull_request_commit_sha,
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
        git_model_version,
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
                True,
                regression_model_info,
                git_model_version,
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
            assert "networkEgressPolicy" in keys

    def test_minimal_payload_setup_for_custom_model_version_creation(
        self, minimal_regression_model_info, git_model_version
    ):
        """A case to test a minimal payload setup when creating a custom model version."""

        payload, file_objs = DrClient._setup_payload_for_custom_model_version_creation(
            True,
            minimal_regression_model_info,
            git_model_version,
            None,
            None,
            base_env_id=str(ObjectId()),
        )
        self._validate_mandatory_attributes_for_regression_model_version(
            payload, optional_exist=False
        )
        assert not file_objs

    @pytest.mark.parametrize("is_major_update", [True, False], ids=["major", "minor"])
    @responses.activate
    def test_create_custom_model_version_success(
        self,
        dr_client,
        custom_model_id,
        is_major_update,
        regression_model_info,
        custom_models_version_url_factory,
        git_model_version,
        regression_model_version_response,
    ):
        """A case to test a successful custom model version creation."""

        url = custom_models_version_url_factory()
        responses.add(responses.POST, url, json=regression_model_version_response, status=201)
        version_id = dr_client.create_custom_model_version(
            custom_model_id, is_major_update, regression_model_info, git_model_version
        )
        assert version_id is not None

    @responses.activate
    def test_create_custom_model_version_failure(
        self,
        dr_client,
        custom_model_id,
        regression_model_info,
        custom_models_version_url_factory,
        git_model_version,
    ):
        """A case to test a failure in creating a custom model version."""

        status_code = 422
        url = custom_models_version_url_factory()
        responses.add(responses.POST, url, json={}, status=status_code)
        with pytest.raises(DataRobotClientError) as ex:
            dr_client.create_custom_model_version(
                custom_model_id, True, regression_model_info, git_model_version
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

        def test_partial_custom_model_testing_configuration(self):
            """A case to test a full configuration of custom model testing."""

            partial_configuration = {
                ModelSchema.NULL_VALUE_IMPUTATION_KEY: {
                    ModelSchema.CHECK_ENABLED_KEY: True,
                    ModelSchema.BLOCK_DEPLOYMENT_IF_FAILS_KEY: True,
                }
            }

            configuration = DrClient._build_tests_configuration(partial_configuration)
            assert (
                DrApiCustomModelChecks.to_dr_attr(ModelSchema.NULL_VALUE_IMPUTATION_KEY)
                in configuration
            )
            # The following checks are silently added
            for check in ["longRunningService", "errorCheck"]:
                assert check in configuration

        def test_full_custom_model_testing_configuration(self, mock_full_custom_model_checks):
            """A case to test a full configuration of custom model testing."""

            assert mock_full_custom_model_checks.keys() == DrApiCustomModelChecks.MAPPING.keys()
            configuration = DrClient._build_tests_configuration(mock_full_custom_model_checks)
            for check in DrApiCustomModelChecks.MAPPING:
                assert DrApiCustomModelChecks.to_dr_attr(check) in configuration
            for check in ["longRunningService", "errorCheck"]:
                assert check in configuration

        def test_full_custom_model_testing_configuration_with_all_disabled_checks(
            self, mock_full_custom_model_checks, default_checks_config
        ):
            """
            A case to test a full custom model testing configuration, when all the checks are
            disabled.
            """

            assert mock_full_custom_model_checks.keys() == DrApiCustomModelChecks.MAPPING.keys()
            for _, info in mock_full_custom_model_checks.items():
                info[ModelSchema.CHECK_ENABLED_KEY] = False
            configuration = DrClient._build_tests_configuration(mock_full_custom_model_checks)
            assert configuration == default_checks_config

        def test_full_custom_model_testing_parameters(self, mock_full_custom_model_checks):
            """A case to test a full number of parameters in custom model testing."""

            assert mock_full_custom_model_checks.keys() == DrApiCustomModelChecks.MAPPING.keys()
            parameters = DrClient._build_tests_parameters(mock_full_custom_model_checks)
            dr_stability_check_key = DrApiCustomModelChecks.to_dr_attr(ModelSchema.STABILITY_KEY)

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
                assert DrApiCustomModelChecks.to_dr_attr(check) in parameters

        def test_full_custom_model_testing_parameters_with_all_disabled_checks(
            self, mock_full_custom_model_checks
        ):
            """
            A case to test a full number of testing parameters, when all the checks are disabled.
            """

            assert mock_full_custom_model_checks.keys() == DrApiCustomModelChecks.MAPPING.keys()
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


class TestRegisteredModels:
    """Registered models tests."""

    @pytest.fixture
    def registered_model_response_mock(self, paginated_url_factory):
        """Return existing registered model"""
        with responses.RequestsMock():
            registered_model = {
                "id": "existing_registered_model_id",
                "name": "existing_registered_model",
            }

            params = {"search": registered_model["name"]}
            mock_single_page_response(
                paginated_url_factory(DrClient.REGISTERED_MODELS_LIST_ROUTE),
                entities=[registered_model],
                match=[matchers.query_param_matcher(params)],
            )

            yield registered_model

    @responses.activate
    def test_create_new_registered_model(self, dr_client, paginated_url_factory):
        """Test creating new registered model"""
        params = {"search": "non_existent_registered_model"}
        mock_single_page_response(
            paginated_url_factory(DrClient.REGISTERED_MODELS_LIST_ROUTE),
            entities=[],
            match=[matchers.query_param_matcher(params)],
        )

        create_model_package_payload = {
            "customModelVersionId": "custom_model_version_id",
            "registeredModelName": "non_existent_registered_model",
        }

        responses.post(
            url=paginated_url_factory(DrClient.MODEL_PACKAGES_CREATE_ROUTE),
            match=[matchers.json_params_matcher(create_model_package_payload)],
            json={"id": "new_registered_model_id"},
            status=201,
        )

        registered_model_version = dr_client.create_or_update_registered_model(
            "custom_model_version_id", "non_existent_registered_model"
        )

        assert registered_model_version == "new_registered_model_id"

    @responses.activate
    def test_update_existing_registered_model(
        self,
        dr_client,
        paginated_url_factory,
        custom_model_version_id,
        registered_model_response_mock,
    ):
        """Update existing registered model by creating new version."""

        mock_single_page_response(
            paginated_url_factory(
                DrClient.REGISTERED_MODELS_VERSIONS_ROUTE.format(
                    registered_model_id=registered_model_response_mock["id"]
                )
            ),
            entities=[],
        )

        create_model_package_payload = {
            "customModelVersionId": custom_model_version_id,
            "registeredModelId": registered_model_response_mock["id"],
        }
        new_registered_model_id = "new_registered_model_id"
        responses.post(
            url=paginated_url_factory(DrClient.MODEL_PACKAGES_CREATE_ROUTE),
            match=[matchers.json_params_matcher(create_model_package_payload)],
            json={"id": new_registered_model_id},
            status=201,
        )
        registered_model_version = dr_client.create_or_update_registered_model(
            custom_model_version_id,
            registered_model_response_mock["name"],
        )

        assert registered_model_version == new_registered_model_id

    @responses.activate
    def test_version_already_registered(
        self,
        dr_client,
        paginated_url_factory,
        custom_model_version_id,
        registered_model_response_mock,
    ):
        """Existing registered model that already contains this version should do nothing."""
        registered_model_version_id = "registered_model_version_id"
        mock_single_page_response(
            paginated_url_factory(
                DrClient.REGISTERED_MODELS_VERSIONS_ROUTE.format(
                    registered_model_id=registered_model_response_mock["id"]
                )
            ),
            entities=[
                {
                    "id": registered_model_version_id,
                    "modelId": custom_model_version_id,
                }
            ],
        )

        registered_model_version = dr_client.create_or_update_registered_model(
            custom_model_version_id,
            registered_model_response_mock["name"],
        )

        assert registered_model_version == registered_model_version_id

    @pytest.mark.parametrize("is_already_global", [True, False])
    @responses.activate
    def test_update_global(self, dr_client, paginated_url_factory, is_already_global):
        """Test setting registered model as global"""

        registered_model = {
            "id": "registered_model_id",
            "name": "registered_model_name",
            "isGlobal": is_already_global,
        }

        params = {"search": registered_model["name"]}
        mock_single_page_response(
            paginated_url_factory(DrClient.REGISTERED_MODELS_LIST_ROUTE),
            entities=[registered_model],
            match=[matchers.query_param_matcher(params)],
        )

        patch_mock = responses.patch(
            url=paginated_url_factory(
                DrClient.REGISTERED_MODEL_ROUTE.format(registered_model_id=registered_model["id"])
            ),
            status=200,
        )

        dr_client.update_registered_model(registered_model["name"], None, is_global=True)

        assert patch_mock.call_count == 0 if is_already_global else 1

    @pytest.mark.parametrize("existing_description", ["same", "different"])
    @responses.activate
    def test_update_description(self, dr_client, paginated_url_factory, existing_description):
        """Test setting registered model as global"""

        registered_model = {
            "id": "registered_model_id",
            "name": "registered_model_name",
            "description": existing_description,
        }

        params = {"search": registered_model["name"]}
        mock_single_page_response(
            paginated_url_factory(DrClient.REGISTERED_MODELS_LIST_ROUTE),
            entities=[registered_model],
            match=[matchers.query_param_matcher(params)],
        )

        patch_mock = responses.patch(
            url=paginated_url_factory(
                DrClient.REGISTERED_MODEL_ROUTE.format(registered_model_id=registered_model["id"])
            ),
            status=200,
        )

        dr_client.update_registered_model(registered_model["name"], "same", is_global=None)

        assert patch_mock.call_count == 0 if existing_description == "same" else 1

    @responses.activate
    def test_update_non_existent(self, dr_client, paginated_url_factory):
        """Test that non existent registered model raises error"""

        mock_single_page_response(
            paginated_url_factory(DrClient.REGISTERED_MODELS_LIST_ROUTE),
            entities=[],
        )

        with pytest.raises(DataRobotClientError):
            dr_client.update_registered_model(
                "non_existent_registered_model", "description", is_global=True
            )

    @responses.activate
    def test_update_error(self, dr_client, paginated_url_factory):
        """Test updating registered model fails"""

        registered_model = {
            "id": "registered_model_id",
            "name": "registered_model_name",
            "description": "model_description",
        }

        params = {"search": registered_model["name"]}
        mock_single_page_response(
            paginated_url_factory(DrClient.REGISTERED_MODELS_LIST_ROUTE),
            entities=[registered_model],
            match=[matchers.query_param_matcher(params)],
        )

        responses.patch(
            url=paginated_url_factory(
                DrClient.REGISTERED_MODEL_ROUTE.format(registered_model_id=registered_model["id"])
            ),
            status=500,
        )

        with pytest.raises(DataRobotClientError):
            dr_client.update_registered_model(
                registered_model["name"], "description", is_global=True
            )


class TestDeploymentRoutes(SharedRouteTests):
    """Contains unit-tests to test the DataRobot deployment routes."""

    def _fetch_entities(self, dr_client):
        return dr_client.fetch_deployments()

    @pytest.fixture
    def deployment_response_factory(self):
        """A factory fixture to create a deployment response."""

        def _inner(deployment_id):
            return {
                "id": deployment_id,
                "userProvidedId": Namespace.namespaced(f"user-provided-id-{deployment_id}"),
            }

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

        self._test_fetch_entities_success(
            dr_client,
            total_num_deployments,
            num_deployments_in_page,
            deployments_url_factory,
            deployment_response_factory,
        )

    @responses.activate
    def test_fetch_deployments_with_multiple_namespaces_success(
        self,
        dr_client,
        deployments_url_factory,
        deployment_response_factory,
    ):
        """A case to test a successful deployments retrieval with multiple namespaces."""

        self._test_fetch_entities_with_multiple_namespace_success(
            deployments_url_factory,
            deployment_response_factory,
            dr_client.fetch_deployments,
        )


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
                "desired": DeploymentSchema.IMPORTANCE_MODERATE,
                "actual": DeploymentSchema.IMPORTANCE_LOW,
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
        importance_value = DeploymentSchema.IMPORTANCE_LOW
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
                "desired": DeploymentSchema.IMPORTANCE_LOW,
                "actual": DeploymentSchema.IMPORTANCE_LOW,
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
                "actual": DeploymentSchema.IMPORTANCE_LOW,
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
                "actual": DeploymentSchema.IMPORTANCE_LOW,
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


class TestSetupSegmentAnalysis:
    """Contains unit-test for the method to setup segment analaysis."""

    @pytest.mark.parametrize(
        "desired_enabled, actual_enabled",
        [(None, None), (False, False), (True, True)],
        ids=["both-none", "both-false", "both-true"],
    )
    def test_desired_equals_actual__no_attributes(self, desired_enabled, actual_enabled):
        """
        Test the expected payload if segment analysis enablement is the same for desired and
        actual settings.
        """

        desired_settings = {DeploymentSchema.SETTINGS_SECTION_KEY: {}}
        if desired_enabled is not None:
            desired_settings[DeploymentSchema.SETTINGS_SECTION_KEY] = {
                DeploymentSchema.SEGMENT_ANALYSIS_KEY: {
                    DeploymentSchema.ENABLE_SEGMENT_ANALYSIS_KEY: desired_enabled,
                }
            }
        deployment_info = DeploymentInfo("dummy.yaml", desired_settings)
        if actual_enabled is None:
            actual_settings = None
        else:
            actual_settings = {"segmentAnalysis": {"enabled": actual_enabled}}

        segmented_analysis_payload = DrClient._setup_segmented_analysis(
            deployment_info, actual_settings
        )
        assert not segmented_analysis_payload

    def test_desired_exists_with_attributes__no_actual_settings(self):
        """
        Test the expected payload if segment analysis settings are fully configured while
        actual(remote) configuration is not set.
        """

        desired_settings = {
            DeploymentSchema.SETTINGS_SECTION_KEY: {
                DeploymentSchema.SEGMENT_ANALYSIS_KEY: {
                    DeploymentSchema.ENABLE_SEGMENT_ANALYSIS_KEY: True,
                    DeploymentSchema.SEGMENT_ANALYSIS_ATTRIBUTES_KEY: ["attr-1", "attr-2"],
                },
            }
        }
        deployment_info = DeploymentInfo("dummy.yaml", desired_settings)
        segmented_analysis_payload = DrClient._setup_segmented_analysis(
            deployment_info, actual_settings=None
        )
        expected_attrs = desired_settings[DeploymentSchema.SETTINGS_SECTION_KEY][
            DeploymentSchema.SEGMENT_ANALYSIS_KEY
        ][DeploymentSchema.SEGMENT_ANALYSIS_ATTRIBUTES_KEY]
        assert segmented_analysis_payload == {"enabled": True, "attributes": expected_attrs}

    def test_desired_exists__no_actual_settings__no_attributes(self):
        """
        Test the expected payload if segment analysis settings was only enabled without
        actual(remote) configuration.
        """

        desired_settings = {
            DeploymentSchema.SETTINGS_SECTION_KEY: {
                DeploymentSchema.SEGMENT_ANALYSIS_KEY: {
                    DeploymentSchema.ENABLE_SEGMENT_ANALYSIS_KEY: True,
                }
            }
        }
        deployment_info = DeploymentInfo("dummy.yaml", desired_settings)
        segmented_analysis_payload = DrClient._setup_segmented_analysis(
            deployment_info, actual_settings=None
        )
        assert segmented_analysis_payload == {"enabled": True}

    @pytest.mark.parametrize(
        "orig_segmented_analysis_attrs", [None, ["orig-attr-1", "oring-attr-2"]]
    )
    def test_desired_and_actual_settings_enabled__new_attributes(
        self, orig_segmented_analysis_attrs
    ):
        """
        Test the expected payload if segment analysis settings exist with new attributes, which
        different from the actual (remote) settings.
        """

        new_segment_analysis_attrs = ["new-attr-1", "new-attr-2"]
        desired_settings = {
            DeploymentSchema.SETTINGS_SECTION_KEY: {
                DeploymentSchema.SEGMENT_ANALYSIS_KEY: {
                    DeploymentSchema.ENABLE_SEGMENT_ANALYSIS_KEY: True,
                    DeploymentSchema.SEGMENT_ANALYSIS_ATTRIBUTES_KEY: new_segment_analysis_attrs,
                }
            }
        }
        deployment_info = DeploymentInfo("dummy.yaml", desired_settings)
        actual_settings = {"segmentAnalysis": {"enabled": True}}
        if orig_segmented_analysis_attrs:
            actual_settings["segmentAnalysis"]["attributes"] = orig_segmented_analysis_attrs
        segmented_analysis_payload = DrClient._setup_segmented_analysis(
            deployment_info, actual_settings
        )
        assert segmented_analysis_payload == {
            "enabled": True,
            "attributes": new_segment_analysis_attrs,
        }


class TestModelReplacementPayloadConstruction:
    """Contains unit-test for the method to set up model replacement payload in a deployment."""

    def test_default_replacement_reason(self, minimal_regression_model_info):
        """
        Test a case where user does not provide a replacement reason in the model's definition file.
        """

        payload = DrClient._setup_model_replacement_payload(
            minimal_regression_model_info, ObjectId()
        )
        assert payload["reason"] == ModelSchema.MODEL_REPLACEMENT_REASON_OTHER

    def test_replacement_reason_success(self, minimal_regression_model_info):
        """Test all valid model replacement values in a model's definition file."""

        model_replacement_values = None
        schema_version_section = ModelSchema.MODEL_SCHEMA.schema[ModelSchema.VERSION_KEY]
        for key, value in schema_version_section.items():
            if (
                isinstance(key, schema.Optional)
                and key.schema == ModelSchema.MODEL_REPLACEMENT_REASON_KEY
            ):
                model_replacement_values = value.args
        assert model_replacement_values is not None
        for replacement_reason in model_replacement_values:
            minimal_regression_model_info.set_value(
                ModelSchema.VERSION_KEY,
                ModelSchema.MODEL_REPLACEMENT_REASON_KEY,
                value=replacement_reason,
            )
            payload = DrClient._setup_model_replacement_payload(
                minimal_regression_model_info, ObjectId()
            )
            assert payload["reason"] == replacement_reason
