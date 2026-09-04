"""Versioned routing evaluation: dataset contracts, metrics, and release gate.

Nothing here talks to a provider. `scripts/evaluate_routing.py` owns the live
calls and writes a `RoutingEvaluationReport`; everything in this package is
deterministic so a release decision can be re-derived from a stored report
without re-running the model.
"""

from __future__ import annotations

__all__ = ["contracts", "metrics", "release_gate"]
