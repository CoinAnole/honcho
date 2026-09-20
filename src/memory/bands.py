"""Locked distance cutoffs. Constants, not settings.

Widening them is a product decision, not a deployment knob.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Final

SAME_CLAIM_MAX: Final[float] = 0.05
CANDIDATE_MAX: Final[float] = 0.15


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
