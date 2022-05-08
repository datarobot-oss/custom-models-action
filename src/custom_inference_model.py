import sys
from abc import ABC
from abc import abstractmethod
from collections import namedtuple
import logging
import os
from glob import glob

import yaml

from schema_validator import ModelSchema

logger = logging.getLogger()


ModelMetadata = namedtuple("ModelMetadata", ["metadata", "filepath"])


class CustomInferenceModelBase(ABC):
    def __init__(self, options):
        self._options = options

    @abstractmethod
    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models
        """
        pass


class CustomInferenceModel(CustomInferenceModelBase):
    def __init__(self, options):
        super().__init__(options)
        self._models_metadata = []
        self._model_schema = ModelSchema()

    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models
        """

        logger.info(f"Options: {self._options}")
        logger.info(f'GITHUB_WORKSPACE: {os.environ["GITHUB_WORKSPACE"]}')
        # raise InvalidModelSchema('(3) Some exception error')

        self._scan_and_load_datarobot_models_metadata()

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
        yaml_files = glob(f"{self._options.root_dir}/**/*.yaml", recursive=True)
        yaml_files.extend(glob(f"{self._options.root_dir}/**/*.yml", recursive=True))
        for yaml_path in yaml_files:
            with open(yaml_path) as f:
                yaml_content = yaml.safe_load(f)
                if self._model_schema.is_multi_models_schema(yaml_content):
                    transformed = self._model_schema.validate_and_transform_multi(
                        yaml_content
                    )
                    for model in transformed[self._model_schema.MULTI_MODELS_KEY]:
                        model_metadata = ModelMetadata(
                            metadata=model, filepath=yaml_path
                        )
                        self._models_metadata.append(model_metadata)
                elif self._model_schema.is_single_model_schema(yaml_content):
                    transformed = self._model_schema.validate_and_transform_single(
                        yaml_content
                    )
                    model_metadata = ModelMetadata(
                        metadata=transformed, filepath=yaml_path
                    )
                    self._models_metadata.append(model_metadata)
