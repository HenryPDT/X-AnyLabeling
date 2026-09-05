"""Shared model-type normalization (Qt-free).

Single source of truth for lenient UI dispatch comparisons such as
``grounding_dino`` vs ``groundingdino`` vs ``grounding-dino``.
Backends keep strict exact matching; UI uses this helper so historical
variants do not silently miss.
"""

from __future__ import annotations

from typing import Any


def normalize_model_type(model_type: Any) -> str:
    """Return lowercase alphanumeric-only model type for comparison."""
    if not model_type:
        return ""
    return "".join(ch for ch in str(model_type).lower() if ch.isalnum())
