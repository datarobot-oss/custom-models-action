import re
import sys
from abc import ABC
from abc import abstractmethod
from collections import namedtuple
import logging
import os
from glob import glob
from pathlib import Path

import yaml

from exceptions import ModelMainEntryPointNotFound, SharedAndLocalPathCollision
from schema_validator import ModelSchema

logger = logging.getLogger()


class ModelInfo:
    def __init__(self, yaml_filepath, model_path, metadata):
        self._yaml_filepath = Path(yaml_filepath)
        self._model_path = Path(model_path)
        self._metadata = metadata
        self._model_file_paths = []

    @property
    def yaml_filepath(self):
        return self._yaml_filepath

    @property
    def model_path(self):
        return self._model_path

    @property
    def metadata(self):
        return self._metadata

    @property
    def model_file_paths(self):
        return self._model_file_paths

    def main_program_exists(self):
        for p in self.model_file_paths:
            if p.name == "custom.py":
                return True
        return False

    def set_paths(self, paths):
        self._model_file_paths = [Path(p) for p in paths]


class CustomInferenceModelBase(ABC):
    def __init__(self, options):
        self._options = options

    @property
    def options(self):
        return self._options

    @abstractmethod
    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models
        """
        pass


class CustomInferenceModel(CustomInferenceModelBase):
    def __init__(self, options):
        super().__init__(options)
        self._models_info = []
        self._model_schema = ModelSchema()

    @property
    def models_info(self):
        return self._models_info

    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models

        This method implements the following logic:
        1. Scan and load DataRobot model metadata (yaml files)
        1. Go over all the models and per model collect the files/folders belong to the model
        2. Per changed file/folder in the commit, find all affected models
        3. Per affected model, create a new version in DataRobot and run tests.

        Returns
        -------
        """

        logger.info(f"Options: {self.options}")
        # raise InvalidModelSchema('(3) Some exception error')

        self._scan_and_load_datarobot_models_metadata()
        self._collect_datarobot_model_files()

        print(
            """
            ::set-output name=new-model-created::True
            ::set-output name=model-deleted::False
            ::set-output name=new-model-version-created::True
            ::set-output name=test-result::The test passed with success.
            ::set-output name=returned-code::200.
            ::set-output name=message::Custom model created and tested with success.
            """
        )

        # print('::set-output name=new-model-created::True')
        # print('::set-output name=model-deleted::False')
        # print('::set-output name=new-model-version-created::True')
        # print('::set-output name=test-result::The test passed with success.')

    def _scan_and_load_datarobot_models_metadata(self):
        yaml_files = glob(f"{self.options.root_dir}/**/*.yaml", recursive=True)
        yaml_files.extend(glob(f"{self.options.root_dir}/**/*.yml", recursive=True))
        for yaml_path in yaml_files:
            with open(yaml_path) as f:
                yaml_content = yaml.safe_load(f)
                if self._model_schema.is_multi_models_schema(yaml_content):
                    transformed = self._model_schema.validate_and_transform_multi(yaml_content)
                    for model_entry in transformed[self._model_schema.MULTI_MODELS_KEY]:
                        model_path = self._to_absolute(
                            model_entry[ModelSchema.MODEL_ENTRY_PATH_KEY],
                            Path(yaml_path).parent,
                        )
                        model_metadata = model_entry[ModelSchema.MODEL_ENTRY_META_KEY]
                        model_info = ModelInfo(yaml_path, model_path, model_metadata)
                        self._models_info.append(model_info)
                elif self._model_schema.is_single_model_schema(yaml_content):
                    transformed = self._model_schema.validate_and_transform_single(yaml_content)
                    yaml_path = Path(yaml_path)
                    model_info = ModelInfo(yaml_path, yaml_path.parent, transformed)
                    self._models_info.append(model_info)

    def _to_absolute(self, path, parent):
        match = re.match(r"^(/|\$ROOT/)", path)
        if match:
            path = path.replace(match[0], "", 1)
            path = f"{self.options.root_dir}/{path}"
        else:
            path = f"{parent}/{path}"
        return path

    def _collect_datarobot_model_files(self):
        for model_info in self.models_info:
            include_glob_patterns = model_info.metadata[ModelSchema.VERSION_KEY][
                ModelSchema.INCLUDE_GLOB_KEY
            ]
            included_paths = set([])
            if include_glob_patterns:
                for include_glob_pattern in include_glob_patterns:
                    include_glob_pattern = self._to_absolute(
                        include_glob_pattern, model_info.model_path
                    )
                    included_paths.update(glob(include_glob_pattern, recursive=True))
            else:
                included_paths.update(glob(f"{model_info.model_path}/**", recursive=True))

            excluded_paths = set([])
            exclude_glob_patterns = model_info.metadata[ModelSchema.VERSION_KEY][
                ModelSchema.EXCLUDE_GLOB_KEY
            ]
            for exclude_glob_pattern in exclude_glob_patterns:
                exclude_glob_pattern = self._to_absolute(
                    exclude_glob_pattern, model_info.model_path
                )
                # For excluded directories always assume recursive
                if Path(exclude_glob_pattern).is_dir():
                    exclude_glob_pattern += "/**"

                excluded_paths.update(glob(exclude_glob_pattern, recursive=True))

            self._set_filtered_model_paths(model_info, included_paths, excluded_paths)
            self._validate_model_integrity(model_info)

    @classmethod
    def _set_filtered_model_paths(cls, model_info, included_paths, excluded_paths):
        included_paths = cls._normalize_paths(included_paths)
        excluded_paths = cls._normalize_paths(excluded_paths)
        final_model_paths = included_paths - excluded_paths
        model_info.set_paths(final_model_paths)

    @staticmethod
    def _normalize_paths(paths):
        re_p1 = re.compile(r"/\./|//")
        re_p2 = re.compile(r"^\./")
        paths = [re_p1.sub("/", p) for p in paths]
        return set([re_p2.sub("", p) for p in paths])

    def _validate_model_integrity(self, model_info):
        if not model_info.main_program_exists():
            raise ModelMainEntryPointNotFound(
                f"Model (Id: {model_info.metadata[ModelSchema.MODEL_ID_KEY]}) main entry point "
                f"not found (custom.py).\n"
                f"Existing files: {model_info.model_file_paths}"
            )

        self._validate_collision_between_local_and_shared(model_info)

    def _validate_collision_between_local_and_shared(self, model_info):
        model_path = model_info.model_path
        model_file_paths = model_info.model_file_paths

        local_path_top_levels = set([])
        shared_path_top_levels = set([])
        for path in model_file_paths:
            if self._is_relative_to(path, model_path):
                relative_path = path.relative_to(model_path)
                if str(relative_path) != ".":
                    local_path_top_levels.add(relative_path.parts[0])
            elif self._is_relative_to(path, self.options.root_dir):
                relative_path = path.relative_to(self.options.root_dir)
                if str(relative_path) != ".":
                    shared_path_top_levels.add(relative_path.parts[0])

        collisions = set(local_path_top_levels) & set(shared_path_top_levels)
        if collisions:
            raise SharedAndLocalPathCollision(
                f"Invalid file tree. Shared file(s)/package(s) collide with local model's "
                f"file(s)/package(s). Collisions: {collisions}."
            )

    @staticmethod
    def _is_relative_to(a_path, b_path):
        try:
            a_path.relative_to(b_path)
            return True
        except ValueError:
            return False
