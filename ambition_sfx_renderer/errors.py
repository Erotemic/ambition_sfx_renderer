"""Exceptions used by the renderer."""


class SfxRenderError(RuntimeError):
    """Base class for renderer failures with user-facing messages."""


class BackendUnavailableError(SfxRenderError):
    """Raised when an optional audio backend is not installed or cannot initialize."""


class SchemaError(SfxRenderError):
    """Raised when a YAML cue is missing required fields or has invalid values."""
