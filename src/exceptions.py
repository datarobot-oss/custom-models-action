class GenericException(Exception):
    def __init__(self, msg, code=-1):
        super().__init__(msg)
        self.code = code


class InvalidModelSchema(GenericException):
    pass
