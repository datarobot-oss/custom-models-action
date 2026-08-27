# tests/functional/test_idempotent.py
"""
Functional tests for idempotent custom model creation.
"""
# pylint: disable=redefined-outer-name

from unittest.mock import Mock

import pytest

from common.namepsace import Namespace
from model_info import ModelInfo
from schema_validator import ModelSchema
from tests.conftest import unique_str
from tests.functional.conftest import webserver_accessible

# Constants for test data
VALID_SHA = "0123456789abcdef0123456789abcdef01234567"
VALID_URL = f"https://github.com/user/repo/commit/{VALID_SHA}"


@pytest.fixture
def mock_git_version():
    """Provides a properly configured GitModelVersion mock."""
    mock_git = Mock()
    mock_git.ref_name = "main"
    mock_git.main_branch_commit_sha = VALID_SHA
    mock_git.pull_request_commit_sha = None
    mock_git.commit_url = VALID_URL
    return mock_git


@pytest.fixture
def unique_model_info():
    """Generates a ModelInfo object with a guaranteed unique ID."""
    model_id = unique_str()  # Generates a random string (e.g. uuid)

    info_data = {
        ModelSchema.MODEL_ID_KEY: model_id,
        ModelSchema.TARGET_TYPE_KEY: "Regression",
        ModelSchema.SETTINGS_SECTION_KEY: {
            ModelSchema.NAME_KEY: f"Idempotency Test {model_id}",
            ModelSchema.TARGET_NAME_KEY: "my_target_column",
        },
        ModelSchema.VERSION_KEY: {
            ModelSchema.INCLUDE_GLOB_KEY: ["*.py"],
        },
    }
    return ModelInfo("/tmp/dummy.yaml", "/tmp/dummy", info_data)


@pytest.mark.skipif(not webserver_accessible(), reason="DataRobot webserver is not accessible.")
def test_create_custom_model_idempotent(dr_client, unique_model_info, mock_git_version):
    """
    Verifies that create_custom_model returns the existing entity
    if called twice with the same ID, rather than raising an error.
    """
    created_model = None

    try:
        # 1. First Call: Create the model
        created_model = dr_client.create_custom_model(unique_model_info, mock_git_version)

        # Validation 1: Ensure it was created correctly
        assert created_model is not None

        # This confirms that the test env and client are using the same namespace logic.
        expected_full_id = Namespace.namespaced(unique_model_info.user_provided_id)
        assert created_model["userProvidedId"] == expected_full_id

        # 2. Second Call: Attempt to create again (Idempotency check)
        existing_model = dr_client.create_custom_model(unique_model_info, mock_git_version)

        # Validation 2: Ensure we got the SAME model back
        assert existing_model is not None
        assert existing_model["id"] == created_model["id"]
        assert existing_model["userProvidedId"] == created_model["userProvidedId"]

    finally:
        # 3. Teardown: Guaranteed cleanup even if assertions fail
        if created_model:
            print(f"Cleaning up model {created_model['id']}")
            try:
                dr_client.delete_custom_model_by_model_id(created_model["id"])
            # pylint: disable=broad-exception-caught
            except Exception as e:
                print(f"Warning: Failed to clean up model: {e}")
