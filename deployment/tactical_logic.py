"""
tactical_logic.py
=================
Backwards-compatibility alias. Use deployment.anomaly_logic in new code.
"""

try:
    from .anomaly_logic import AnomalyRegionLogic, TacticalDecisionLogic
except ImportError:
    from anomaly_logic import AnomalyRegionLogic, TacticalDecisionLogic

__all__ = ["AnomalyRegionLogic", "TacticalDecisionLogic"]
