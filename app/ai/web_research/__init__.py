"""Canonical provider-neutral web research."""

from .contracts import (
    ImageCandidateRecord,
    ProviderImageCandidate,
    ProviderSource,
    ResearchFailure,
    ResearchRequest,
    ResearchScope,
    SourceRecord,
    WebEvidenceBundle,
)
from .policy import ResearchLimits
from .source_registry import SourceRegistry, canonicalize_public_url

__all__ = [
    "ImageCandidateRecord",
    "ProviderImageCandidate",
    "ProviderSource",
    "ResearchFailure",
    "ResearchLimits",
    "ResearchRequest",
    "ResearchScope",
    "SourceRecord",
    "SourceRegistry",
    "WebEvidenceBundle",
    "canonicalize_public_url",
]
