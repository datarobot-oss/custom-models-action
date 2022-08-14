#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A module that contains general string operations.
"""


class StringUtil:
    """A string utility class."""

    @staticmethod
    def slash_suffix(s):
        """Add a suffix slash if not exists."""

        return s if s.endswith("/") else f"{s}/"
