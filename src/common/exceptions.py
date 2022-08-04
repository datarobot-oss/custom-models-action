class GenericException(Exception):
    def __init__(self, msg, code=-1):
        super().__init__(msg)
        self.code = code


class InvalidSchema(GenericException):
    def __init__(self, msg, code=-1):
        super().__init__(msg.split("\n")[-1], code)


class InvalidModelSchema(InvalidSchema):
    pass


class InvalidDeploymentSchema(InvalidSchema):
    pass


class UnexpectedType(GenericException):
    pass


class UnexpectedResult(GenericException):
    pass


class UnexpectedInput(GenericException):
    pass


class InvalidMemoryValue(GenericException):
    pass


class ModelMainEntryPointNotFound(GenericException):
    pass


class PathOutsideTheRepository(GenericException):
    pass


class SharedAndLocalPathCollision(GenericException):
    pass


class UnInitializedGitTool(GenericException):
    pass


class NoCommonAncestor(GenericException):
    pass


class NoValidAncestor(GenericException):
    pass


class HttpRequesterException(GenericException):
    pass


class DataRobotClientError(GenericException):
    pass


class NonMergeCommitError(GenericException):
    pass


class IllegalModelDeletion(GenericException):
    pass


class ModelMetadataAlreadyExists(GenericException):
    pass


class DeploymentMetadataAlreadyExists(GenericException):
    pass


class AssociatedModelNotFound(GenericException):
    pass


class AssociatedModelVersionNotFound(GenericException):
    pass


class UnexpectedNumOfModelVersions(GenericException):
    pass


class TooFewArguments(GenericException):
    pass
