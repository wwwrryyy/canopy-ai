"""
sharp_sync_daemon.py
====================
Backwards-compatibility wrapper for flight telemetry.
Delegates to deployment.flight_telemetry.FlightTelemetrySpooler.
"""

from flight_telemetry import FlightTelemetrySpooler


class SHARPOfflineResilienceManager(FlightTelemetrySpooler):
    """
    Backwards-compatible alias for FlightTelemetrySpooler.
    Preserves existing interfaces while fixing dictionary key mismatches and disk locks.
    """

    def log_threat_offline(self, threat_dict):
        """Deprecated legacy alias. Use record() or log_anomaly_offline()."""
        return self.record(threat_dict)


__all__ = ["SHARPOfflineResilienceManager"]
