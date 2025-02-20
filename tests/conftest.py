#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A configuration test for both functional and unit-tests."""
import os
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from common.github_env import GitHubEnv


def unique_str():
    """Generate a unique 10-chars long string."""

    return f"{random.randint(1, 2 ** 32):010}"


@pytest.fixture
def github_output():
    """
    A fixture to emulate the 'GITHUB_OUTPUT' environment variable, which points to and
    existing output file.
    """

    github_output_env = GitHubEnv.github_output()
    if not github_output_env:
        github_output_filepath = Path("/tmp/github_output")
        with open(github_output_filepath, "w", encoding="utf-8"), patch.dict(
            os.environ, {"GITHUB_OUTPUT": str(github_output_filepath)}
        ):
            yield github_output_filepath

        # Just in case the file was not deleted by a certain test
        if github_output_filepath.is_file():
            github_output_filepath.unlink()
    else:
        yield github_output_env
