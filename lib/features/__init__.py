"""Feature computation.

``lib/features.py`` used to sit next to this package and was shadowed by it.
The feature code now lives in ``core.py`` and is re-exported here, so
``from features import compute_features, FEATURE_NAMES`` keeps working.
"""

from .core import *  # noqa: F403
from .core import FEATURE_NAMES, compute_features  # noqa: F401
