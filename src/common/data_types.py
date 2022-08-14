#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""Contains various data types."""

from dataclasses import dataclass


@dataclass
class DataRobotModel:
    """
    Contains information about a specific model in DataRobot. It holds the model as well as
    the latest version of that model.
    """

    model: dict
    latest_version: dict


@dataclass
class DataRobotDeployment:
    """
    Contains information about a specific deployment in DataRobot. It holds the deployment
    as well as the model version that is used in that deployment.
    """

    deployment: dict
    model_version: dict
