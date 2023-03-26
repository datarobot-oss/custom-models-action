#  Copyright (c) 2023. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

# pylint: disable=too-many-arguments

"""
Functional tests for a single definition YAML file which contains multiple model's definition.
The functional tests are executed against a running DataRobot application. If DataRobot is not
accessible, the functional tests are skipped.
"""

import glob

import pytest
import yaml

from schema_validator import ModelSchema
from tests.functional.conftest import printout
from tests.functional.conftest import run_github_action
from tests.functional.conftest import webserver_accessible


# pylint: disable=too-few-public-methods
@pytest.mark.skipif(not webserver_accessible(), reason="DataRobot webserver is not accessible.")
@pytest.mark.usefixtures("github_output", "cleanup")
class TestMultiModelsOneDefinitionGitHubAction:
    """A class to test multi-models definition in a single metadata YAML file"""

    @pytest.mark.parametrize(
        "is_abs_model_path, model_path_prefix", [(True, "/"), (True, "$ROOT/"), (False, None)]
    )
    def test_e2e_pull_request_event_with_multi_model_definition(
        self,
        dr_client,
        workspace_path,
        git_repo,
        main_branch_name,
        build_repo_for_testing_factory,
        is_abs_model_path,
        model_path_prefix,
    ):
        """
        An end-to-end case to test model deletion by the custom inference model GitHub action
        from a pull-request. The test first creates a PR with a simple change in order to create
        the model in DataRobot. Afterwards, it creates another PR to delete the model definition,
        which should delete the model in DataRobot.
        """

        build_repo_for_testing_factory(
            dedicated_model_definition=False,
            is_absolute_model_path=is_abs_model_path,
            model_path_prefix=model_path_prefix,
        )
        printout("Run custom model GitHub action (push event) ...")
        run_github_action(workspace_path, git_repo, main_branch_name, "push", is_deploy=False)

        printout("Validate after merging ...")

        multi_models_metadata = self._load_multi_models_metadata(workspace_path)
        models = dr_client.fetch_custom_models()

        expected_user_provided_ids = {
            model_entry[ModelSchema.MODEL_ENTRY_META_KEY][ModelSchema.MODEL_ID_KEY]
            for model_entry in multi_models_metadata[ModelSchema.MULTI_MODELS_KEY]
        }
        actual_user_provided_ids = {model.get("userProvidedId") for model in models}
        assert expected_user_provided_ids <= actual_user_provided_ids
        printout("Done")

    @staticmethod
    def _load_multi_models_metadata(workspace_path):
        multi_models_yaml_filepath = glob.glob(str(workspace_path / "**/models.yaml"))
        assert len(multi_models_yaml_filepath) == 1
        multi_models_yaml_filepath = multi_models_yaml_filepath[0]
        with open(multi_models_yaml_filepath, encoding="utf-8") as fd:
            multi_models_metadata = ModelSchema.validate_and_transform_multi(yaml.safe_load(fd))
        return multi_models_metadata
