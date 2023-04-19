#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A module to provide an interface to GitHub environment variables"""

import os
import uuid
from pathlib import Path


class GitHubEnv:
    """
    Retrieve and return information from environment variables in the GitHub runner environment.
    """

    @staticmethod
    def event_name():
        """The event name that triggered the GitHub workflow."""

        return os.environ.get("GITHUB_EVENT_NAME")

    @staticmethod
    def github_sha():
        """The commit SHA that triggered the GitHub workflow"""

        return os.environ.get("GITHUB_SHA")

    @staticmethod
    def github_repository():
        """The owner and repository name from GtiHub"""

        return os.environ.get("GITHUB_REPOSITORY")

    @staticmethod
    def ref_name():
        """The branch or tag name that triggered the GitHub workflow run."""

        return os.environ.get("GITHUB_REF_NAME")

    @classmethod
    def is_pull_request(cls):
        """Whether the event that triggered the GitHub workflow is a pull request."""

        return cls.event_name() == "pull_request"

    @classmethod
    def is_push(cls):
        """Whether the event that triggered the GitHub workflow is a push."""

        return cls.event_name() == "push"

    @staticmethod
    def base_ref():
        """The name of the base ref or target branch of the pull request in a workflow run."""

        return os.environ.get("GITHUB_BASE_REF")

    @staticmethod
    def workspace_path():
        """The default location of the repository when using the checkout action."""

        return Path(os.environ.get("GITHUB_WORKSPACE"))

    @staticmethod
    def repository_id():
        """The ID of the repository."""

        return os.environ.get("GITHUB_REPOSITORY_ID")

    @staticmethod
    def github_output():
        """The GitHub output environment variable that holds the output file."""

        return os.environ.get("GITHUB_OUTPUT")

    @classmethod
    def set_output_param(cls, name, value):
        """Set an output parameter that can be referenced by a follow-up step."""

        is_multiline = False
        if isinstance(value, str):
            value = value.strip()
            is_multiline = "\n" in value

        with open(cls.github_output(), "a", encoding="utf-8") as file:
            if is_multiline:
                eof = uuid.uuid4()
                file.write(f"{name}<<{eof}\n")
                file.write(f"{value}\n")
                file.write(f"{eof}\n")
            else:
                file.write(f"{name}={value}\n")
