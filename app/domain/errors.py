class DomainError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ConfigurationError(DomainError):
    pass


class ProviderError(DomainError):
    def __init__(self, code: str, message: str, *, metadata=None) -> None:
        super().__init__(code, message)
        self.metadata = metadata


class DocumentError(DomainError):
    pass


class RetrievalError(DomainError):
    pass


class WorkflowError(DomainError):
    pass


class CitationError(DomainError):
    pass
