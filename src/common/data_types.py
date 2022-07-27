from dataclasses import dataclass


@dataclass
class DataRobotModel:
    model: dict
    latest_version: dict


@dataclass
class DataRobotDeployment:
    deployment: dict
    model_version: dict
