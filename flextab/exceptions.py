class FlexTabError(Exception):
    pass


class FlexTabValueError(ValueError, FlexTabError):
    pass


class FlexTabInputValidationError(ValueError, FlexTabError):
    pass


class FlexTabModelNotFitted(ValueError, FlexTabError):
    def __init__(self, message='Model not fitted yet'):
        self.message = message
        super().__init__(self.message)
