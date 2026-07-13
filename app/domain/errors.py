class ConfigurationError(Exception):
    code = "configuration_error"


class ProviderError(Exception):
    code = "provider_error"


class DocumentError(Exception):
    code = "document_error"


class RetrievalError(Exception):
    code = "retrieval_error"


class WorkflowError(Exception):
    code = "workflow_error"


class CitationError(Exception):
    code = "citation_error"
