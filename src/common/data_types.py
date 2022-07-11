from dataclasses import dataclass


@dataclass
class FileInfo:
    actual_path: str
    path_under_model: str


@dataclass
class DataRobotModel:
    model: dict
    latest_version: dict


@dataclass
class DataRobotDeployment:
    deployment: dict
    model_version: dict
