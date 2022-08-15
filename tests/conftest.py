#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""A configuration test for both functional and unit-tests."""

import random


def unique_str():
    """Generate a unique 10-chars long string."""

    return f"{random.randint(1, 2 ** 32): 010}"
