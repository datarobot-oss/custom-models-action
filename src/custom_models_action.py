#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
The implementation of the custom models GitHub action. It scans and loads model definitions
from the local source tree, performs validations and then detects which models/deployments
were affected by the last Git action and applies the proper actions in DataRobot application.
"""

import logging

from common.git_tool import GitTool
from common.github_env import GitHubEnv
from deployment_controller import DeploymentController
from model_controller import ModelController

logger = logging.getLogger()


# pylint: disable=too-few-public-methods
class CustomModelsAction:
    """The implementation of the custom models GitHub action."""

    def __init__(self, options):
        self._options = options
        self._repo = GitTool(GitHubEnv.workspace_path())
        self._model_controller = ModelController(options, self._repo)
        self._deployment_controller = None

    @property
    def model_controller(self):
        """A property to return the model controller attribute."""

        return self._model_controller

    @property
    def deployment_controller(self):
        """A property to return the deployment controller attribute."""

        return self._deployment_controller

    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models and deployments.
        The logic takes into account the fact that a model can be deleted along with a deployment
        in the same commit.
        """

        try:
            if not self._prerequisites():
                return

            self.model_controller.scan_and_load_models_metadata()
            self.model_controller.collect_datarobot_model_files()
            self.model_controller.fetch_models_from_datarobot()
            self.model_controller.lookup_affected_models_by_the_current_action()
            self.model_controller.handle_model_changes()

            if not self._options.models_only:
                self._deployment_controller = DeploymentController(
                    self._options, self.model_controller, self._repo
                )

                self.deployment_controller.scan_and_load_deployments_metadata()
                self.deployment_controller.fetch_deployments_from_datarobot()
                self.deployment_controller.validate_deployments_integrity()

            if GitHubEnv.is_push():
                if not self._options.models_only:
                    self.deployment_controller.handle_deployment_changes_or_creation()
                    self.deployment_controller.handle_deleted_deployments()
                # NOTE: a model can be created during a pull request, but it is always
                # deleted during a merge (push event). The reason is to keep integrity between
                # a deployment and the associated model. A model cannot be deleted, unless it is
                # not deployed.
                self.model_controller.handle_deleted_models()

        finally:
            self._save_statistics()

    def _prerequisites(self):
        """Check prerequisites before execution."""

        base_ref = GitHubEnv.base_ref()
        logger.info("GITHUB_BASE_REF: %s.", base_ref)
        if GitHubEnv.is_pull_request() and base_ref != self._options.branch:
            logger.info(
                "Skip custom models action. It is executed only when the referenced "
                "branch is %s. Current ref branch: %s.",
                self._options.branch,
                base_ref,
            )
            return False

        # NOTE: in the case of functional tests, the number of remotes is zero and still it's valid.
        if self._repo.num_remotes() > 1:
            logger.warning(
                "Skip custom models action, because the given repository has more than "
                "one remote configured."
            )
            return False

        if GitHubEnv.is_pull_request():
            # For pull request we assume a merge branch, which contains at least 2 commits
            num_commits = self._repo.num_commits()
            if num_commits < 2:
                logger.warning(
                    "Skip custom models action. The minimum number of commits "
                    "should be 2. Current number is %d.",
                    num_commits,
                )
                return False
        return True

    def _save_statistics(self):
        self.model_controller.save_metrics()
        if self.deployment_controller:
            self.deployment_controller.save_metrics()
