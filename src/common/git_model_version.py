#  Copyright (c) 2023. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A module that holds local git details."""

import logging

from common.git_tool import GitTool
from common.github_env import GitHubEnv

logger = logging.getLogger()


class GitModelVersion:
    """A class that holds local git information."""

    def __init__(self, repo, branch):
        self._ref_name = GitHubEnv.ref_name()
        if GitHubEnv.is_pull_request():
            if repo.num_remotes() == 0:
                # Only to support the functional tests, which do not have a remote repository.
                main_branch = branch
            else:
                # This is the expected path when working against a remote GitHub repository.
                main_branch = f"{repo.remote_name()}/{branch}"
            self._main_branch_commit_sha = repo.merge_base_commit_sha(
                main_branch, GitHubEnv.github_sha()
            )
            self._pull_request_commit_sha = repo.feature_branch_top_commit_sha_of_a_merge_commit(
                GitHubEnv.github_sha()
            )
            self._commit_url = GitTool.GITHUB_COMMIT_URL_PATTERN.format(
                user_and_project=GitHubEnv.github_repository(), sha=self._pull_request_commit_sha
            )
        else:
            self._main_branch_commit_sha = GitHubEnv.github_sha()
            self._pull_request_commit_sha = None
            self._commit_url = GitTool.GITHUB_COMMIT_URL_PATTERN.format(
                user_and_project=GitHubEnv.github_repository(), sha=self._main_branch_commit_sha
            )
        logger.info(
            "GitHub version info. Ref name: %s, commit URL: %s, main branch commit sha: %s, "
            "pull request commit sha: %s",
            self._ref_name,
            self._commit_url,
            self._main_branch_commit_sha,
            self._pull_request_commit_sha,
        )

    @property
    def ref_name(self):
        """The branch or tag name that triggered the GitHub workflow run."""

        return self._ref_name

    @property
    def commit_url(self):
        """The commit URL in GitHub."""

        return self._commit_url

    @property
    def main_branch_commit_sha(self):
        """The commit SHA that triggered the GitHub workflow."""

        return self._main_branch_commit_sha

    @property
    def pull_request_commit_sha(self):
        """
        For pull-requests it is the top commit in a merge branch, which was created from a feature
        branch as part of a pull request. For push events it is None."""

        return self._pull_request_commit_sha
