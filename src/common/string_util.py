#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A module that contains general string operations.
"""


# pylint: disable=too-few-public-methods
class StringUtil:
    """A string utility class."""

    @staticmethod
    def slash_suffix(url_str):
        """Add a suffix slash if not exists."""

        return url_str if url_str.endswith("/") else f"{url_str}/"
