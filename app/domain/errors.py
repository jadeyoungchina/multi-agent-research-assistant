class DomainError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ConfigurationError(DomainError):
    pass


class ProviderError(DomainError):
    pass


class DocumentError(DomainError):
    pass


class RetrievalError(DomainError):
    pass


class WorkflowError(DomainError):
    pass


class CitationError(DomainError):
    pass
