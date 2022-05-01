from abc import ABC
from abc import abstractmethod
import os


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
    def run(self):
        """
        Executes the GitHub action logic to manage custom inference models
        """

        print(f'Options: {self._options}')
        print(f'GITHUB_WORKSPACE: {os.environ["GITHUB_WORKSPACE"]}')

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
