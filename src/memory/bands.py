"""Locked distance cutoffs. Constants, not settings.

Widening them is a product decision, not a deployment knob.
"""

from __future__ import annotations

import math
from datetime import timedelta
from enum import Enum
from typing import Final

SAME_CLAIM_MAX: Final[float] = 0.05
CANDIDATE_MAX: Final[float] = 0.15

# Rank demotion for get_most_derived: score *= 0.5^(age / half_life).
# Product constant, not a deployment knob (Candidate A).
DECAY_HALF_LIFE: Final[timedelta] = timedelta(days=14)


class Band(Enum):
    SAME_CLAIM = "same_claim"
    CANDIDATE = "candidate"
    UNRELATED = "unrelated"


def classify_distance(distance: float) -> Band:
    """Pure. Total over floats. NaN maps to UNRELATED."""
    if math.isnan(distance) or distance > CANDIDATE_MAX:
        return Band.UNRELATED
    if distance <= SAME_CLAIM_MAX:
        return Band.SAME_CLAIM
    return Band.CANDIDATE
