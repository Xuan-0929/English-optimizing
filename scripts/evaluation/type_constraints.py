#!/usr/bin/env python3
"""Type post-processing constraints for grammar-correction outputs."""

from __future__ import annotations

import re
from typing import Optional, Tuple


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _is_marry_with_fix(user_input: str, correction: str) -> bool:
    src = _norm(user_input)
    corr = _norm(correction)
    return "married with" in src and "married with" not in corr and "married " in corr


def _is_reported_speech_will_to_would(user_input: str, correction: str) -> bool:
    src = _norm(user_input)
    corr = _norm(correction)
    # Cover common reported-speech patterns:
    # - "He said ... will ..."
    # - "He said that ... will ..."
    # - "She told me ... will ..."
    has_reporting = re.search(r"\b(said|told)\b", src) is not None
    return has_reporting and " will " in f" {src} " and " would " in f" {corr} "


def apply_type_constraints(
    user_input: str,
    correction: str,
    predicted_type: str,
) -> Tuple[str, Optional[str]]:
    """Return (final_type, applied_rule_name)."""
    if _is_marry_with_fix(user_input, correction):
        return "介词", "married_with_to_transitive_marry"
    if _is_reported_speech_will_to_would(user_input, correction):
        return "时态一致", "reported_speech_will_to_would"
    return predicted_type, None
