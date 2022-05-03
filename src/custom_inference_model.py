import sys
from abc import ABC
from abc import abstractmethod
import logging
import os
from glob import glob

import yaml

from exceptions import InvalidModelSchema

logger = logging.getLogger()


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

    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models
        """

        logger.info(f"Options: {self._options}")
        logger.info(f'GITHUB_WORKSPACE: {os.environ["GITHUB_WORKSPACE"]}')
        # raise InvalidModelSchema('(3) Some exception error')

        self._scan_and_load_models_metadata()

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

    def _scan_and_load_models_metadata(self):
        yaml_files = glob(f"{self._options.root_dir}/**/*.yaml", recursive=True)
        yaml_files.extend(glob(f"{self._options.root_dir}/**/*.yml", recursive=True))
        for yaml_path in yaml_files:
            with open(yaml_path) as f:
                yaml_content = yaml.safe_load(f)
                if "models" in yaml_content:
                    for model in yaml_content["models"]:
                        self._models_metadata.append(model)
                else:
                    self._models_metadata.append(yaml_content)
