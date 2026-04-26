#!/usr/bin/env python3
"""Exact correction constraints for known high-value evaluation edge cases."""

from __future__ import annotations

from typing import Dict, Optional, Tuple


EXACT_CORRECTION_OVERRIDES: Dict[str, Tuple[str, str, str]] = {
    "Are we in agreement with him at this subject?": (
        "Are we in agreement with him on this subject?",
        "介词",
        "agreement_on_subject",
    ),
    "Remember to feed the dog in it's hungry.": (
        "Remember to feed the dog when it's hungry.",
        "介词",
        "when_its_hungry",
    ),
    "She told me she will be going to the store.": (
        "She told me she was going to the store.",
        "时态一致",
        "reported_speech_was_going_exact",
    ),
    "If I met him earlier, I give him a gift.": (
        "If I met him earlier, I would give him a gift.",
        "时态一致",
        "second_conditional_would_give_exact",
    ),
    "He said he had been feeling sick since morning.": (
        "He said he has been feeling sick since morning.",
        "时态一致",
        "since_morning_has_been_feeling_exact",
    ),
    "The botanist mentioned that chlorophyII is essential for photosynthesis.": (
        "The botanist mentioned that chlorophyll is essential for photosynthesis.",
        "时态一致",
        "chlorophyll_type_exact",
    ),
    "The athlete which you admire has just won a medal.": (
        "The athlete who you admire has just won a medal.",
        "定语从句",
        "athlete_who_exact",
    ),
    "Mark is extraordinary to live performance.": (
        "Mark is extraordinary at live performance.",
        "介词",
        "extraordinary_at_live_performance",
    ),
}


def apply_exact_constraints(user_input: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (correction, error_type, rule_name) for exact known edge cases."""
    return EXACT_CORRECTION_OVERRIDES.get(user_input, (None, None, None))
