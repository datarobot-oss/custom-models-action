#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""
A module that contains a definition of a model file path after it is scanned and loaded from the
local source tree.
"""

from enum import Enum
from pathlib import Path

from common.exceptions import PathOutsideTheRepository


class ModelFilePath:
    """
    Holds essential information about a single file that belongs to a certain
    model.
    """

    class RelativeTo(Enum):
        """
        An enum to indicate whether a given file path is relative to the model's root dir or
        to the repository root dir.
        """

        MODEL = 1
        ROOT = 2

    def __init__(self, raw_file_path, model_root_dir, repo_root_dir):
        self._raw_file_path = raw_file_path
        self._filepath = Path(raw_file_path)
        # It is important to have an indication about the path origin and the relation to
        # the model, means whether the given path was originally under the model's root dir
        # or it is supposed to be copied into it. This will help us to detect collisions
        # between paths that exist under the model versus those that are supposed to be copied.
        path_under_model, relative_to = self.get_path_under_model(
            self._filepath, model_root_dir, repo_root_dir
        )
        self._under_model = path_under_model
        self._relative_to = relative_to

    @classmethod
    def get_path_under_model(cls, filepath, model_root_dir, repo_root_dir):
        """
        Returns the relative file path of a given model's file from the model's root directory.

        Parameters
        ----------
        filepath : pathlib.Path
            A model's file path.
        model_root_dir : pathlib.Path
            The model's root directory.
        repo_root_dir : pathlib.Path
            The repository root directory.

        Returns
        -------
        tuple(str, ModelFilePath.RelativeTo),
            The relative path under the model + an enum value to indicate whether the origin
            path is relative to the model's root dir or to the repostiroy root dir.
        """

        try:
            path_under_model = cls._get_path_under_model_for_given_root(filepath, model_root_dir)
            relative_to = cls.RelativeTo.MODEL
        except ValueError:
            try:
                path_under_model = cls._get_path_under_model_for_given_root(filepath, repo_root_dir)
                relative_to = cls.RelativeTo.ROOT
            except ValueError as ex:
                raise PathOutsideTheRepository(
                    f"Model file path is outside the repository: {filepath}"
                ) from ex
        return path_under_model, relative_to

    @staticmethod
    def _get_path_under_model_for_given_root(filepath, root):
        relative_path = filepath.relative_to(root)
        return str(relative_path).replace("../", "")  # Will be copied under the model

    def __str__(self):
        return self.under_model

    @property
    def filepath(self):
        """A single file path that belongs to the model"""
        return self._filepath

    @property
    def name(self):
        """The file name"""
        return self.filepath.name

    @property
    def resolved(self):
        """The full absolute file path, after resolving all soft links"""
        return self.filepath.resolve()

    @property
    def under_model(self):
        """The final relative file path under the model's as it is stored in DataRobot"""
        return self._under_model

    @property
    def relative_to(self):
        """
        An enum value that indicates whether the given file is relative to the model's
        root directory or to the repository root directory.
        """
        return self._relative_to
