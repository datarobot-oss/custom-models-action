#  Copyright (c) 2022. DataRobot, Inc. and its affiliates.
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.

"""Contains definitions of the exception classes that are used in this repository."""


class GenericException(Exception):
    """A generic exception, which is used as the base of all other exception."""

    def __init__(self, msg, code=-1):
        super().__init__(msg)
        self.code = code


class InvalidSchema(GenericException):
    """
    Raised when a schema does not comply with the template. It is also used as the base
    exception for the model and deployment schema exceptions.
    """

    def __init__(self, msg, code=-1):
        super().__init__(msg.split("\n")[-1], code)


# noinspection DuplicatedCode
class InvalidModelSchema(InvalidSchema):
    """Raised when an invalid model schema is detected."""

    pass


class InvalidDeploymentSchema(InvalidSchema):
    """Raised when an invalid deployment schema is detected."""

    pass


class UnexpectedType(GenericException):
    """Raised when an unexpected variable or argument type is detected."""

    pass


class UnexpectedResult(GenericException):
    """Raised when an unexpected result is detected."""

    pass


class UnexpectedInput(GenericException):
    """Raised when unexpected input is detected."""

    pass


class InvalidMemoryValue(GenericException):
    """Raised when an invalid memory value is detected."""

    pass


class ModelMainEntryPointNotFound(GenericException):
    """Raised when a main entry point for a given model was not found."""

    pass


class PathOutsideTheRepository(GenericException):
    """Raised when a path that is supposed to belong to a model is outside the repository."""

    pass


class SharedAndLocalPathCollision(GenericException):
    """
    Raised when a collision is detected between files in a shared location and those under
    the model path.
    """

    pass


class UnInitializedGitTool(GenericException):
    """Raised when trying to use a Git method while the related class was not initialized."""

    pass


class NoCommonAncestor(GenericException):
    """Raised when not common ancestor is found between a merge and a main branches."""

    pass


# noinspection DuplicatedCode
class NoValidAncestor(GenericException):
    """Raised when an associated model's version SHA is not an ancestor in the current tree."""

    pass


class HttpRequesterException(GenericException):
    """Raised when a status URL response is not 200."""

    pass


class DataRobotClientError(GenericException):
    """Raised when a non expected success response is detected by the DataRobot client."""

    pass


class NonMergeCommitError(GenericException):
    """
    Raised when a commit with an invalid expected number of parents is detected for a feature
    branch.
    """

    pass


class IllegalModelDeletion(GenericException):
    """Raised when trying to delete a model while there's an existing deployment with that model."""

    pass


class ModelMetadataAlreadyExists(GenericException):
    """Raised when more than one model metadata is detected with the same Git model ID."""

    pass


class DeploymentMetadataAlreadyExists(GenericException):
    """Raised when more than one deployment metadata is detected with the same Git deployment ID."""

    pass


class AssociatedModelNotFound(GenericException):
    """Raised when validating a deployment for an existing associated model."""

    pass


class AssociatedModelVersionNotFound(GenericException):
    """Raised when validating a deployment for an existing associated model version."""

    pass


class EmptyKey(GenericException):
    """Raised when an invalid empty key is provided to get a value from a metadata."""

    pass
