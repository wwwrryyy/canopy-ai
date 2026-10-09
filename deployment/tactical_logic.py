"""
tactical_logic.py
=================
DEPRECATED: Legacy backwards-compatibility module.
Use deployment.canopy_inspection or deployment.anomaly_logic in modern code.
"""

import warnings

warnings.warn(
    "tactical_logic is deprecated and will be removed in a future release. "
    "Import from deployment.canopy_inspection or deployment.anomaly_logic instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from .anomaly_logic import AnomalyRegionLogic, TacticalDecisionLogic
except ImportError:
    from anomaly_logic import AnomalyRegionLogic, TacticalDecisionLogic

__all__ = ["AnomalyRegionLogic", "TacticalDecisionLogic"]
