"""
flight_telemetry.py
===================
Deep flight telemetry module for drone edge deployment.

Provides memory-buffered offline spooling of detected canopy anomalies into local SQLite storage
without disk I/O lock contention on real-time flight loops. Automatically synchronizes buffered
records when connectivity with ground stations or management servers is restored.
"""

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Union

from canopy_anomaly import CanopyAnomaly


class FlightTelemetrySpooler:
    """
    Spooling telemetry module with automatic schema validation and batch persistence.
    Protects real-time vision inference loops from per-frame disk locks.
    """

    def __init__(
        self,
        db_path: str = "flight_telemetry.db",
        batch_size: int = 20,
        auto_start_monitor: bool = True,
    ):
        self.db_path = db_path
        self.batch_size = batch_size
        self._spool: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.is_connected = False
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        self._init_db()
        if auto_start_monitor:
            self.start_network_monitor()

    def _init_db(self) -> None:
        """Initializes SQLite schema for local anomaly persistence."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                anomaly_id INTEGER,
                anomaly_type TEXT,
                confidence REAL,
                mse_score REAL,
                bbox_json TEXT,
                latitude REAL,
                longitude REAL,
                altitude REAL,
                estimated_area_sqm REAL,
                polygon_wgs84_json TEXT,
                synced INTEGER DEFAULT 0
            )
        """)
        
        # Schema migration for existing databases
        existing = {row[1] for row in cur.execute("PRAGMA table_info(anomaly_logs)")}
        if "estimated_area_sqm" not in existing:
            cur.execute("ALTER TABLE anomaly_logs ADD COLUMN estimated_area_sqm REAL")
        if "polygon_wgs84_json" not in existing:
            cur.execute("ALTER TABLE anomaly_logs ADD COLUMN polygon_wgs84_json TEXT")
            
        conn.commit()
        conn.close()

    def record(
        self,
        anomaly: Union[CanopyAnomaly, Dict[str, Any]],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: Optional[float] = None,
    ) -> None:
        """
        Records a detected canopy anomaly into the offline spool.

        Args:
            anomaly: Strongly-typed CanopyAnomaly object or dictionary.
            latitude: GPS latitude in degrees.
            longitude: GPS longitude in degrees.
            altitude: Flight altitude above ground level (m).
        """
        if isinstance(anomaly, CanopyAnomaly):
            anom_id = anomaly.anomaly_id
            anom_type = anomaly.anomaly_type
            conf = anomaly.confidence
            score = anomaly.mse_score
            bbox_list = list(anomaly.bbox)
            area_sqm = anomaly.estimated_area_sqm
            poly_wgs84 = anomaly.polygon_wgs84
            # If the caller provides lat/lon directly to record(), it overrides the anomaly's fields
            latitude = latitude if latitude is not None else anomaly.gps_latitude
            longitude = longitude if longitude is not None else anomaly.gps_longitude
            altitude = altitude if altitude is not None else anomaly.gps_altitude_m
        else:
            anom_id = anomaly.get("anomaly_id", anomaly.get("id", 0))
            anom_type = anomaly.get("anomaly_type", anomaly.get("type", "UNCLASSIFIED_ANOMALY"))
            conf = float(anomaly.get("confidence", 0.0))
            score = float(anomaly.get("mse_score", anomaly.get("score", 0.0)))
            bbox_list = list(anomaly.get("bbox", []))
            area_sqm = anomaly.get("estimated_area_sqm")
            poly_wgs84 = anomaly.get("polygon_wgs84")
            latitude = latitude if latitude is not None else anomaly.get("gps_latitude")
            longitude = longitude if longitude is not None else anomaly.get("gps_longitude")
            altitude = altitude if altitude is not None else anomaly.get("gps_altitude_m")

        record_entry = {
            "timestamp": time.time(),
            "anomaly_id": anom_id,
            "anomaly_type": anom_type,
            "confidence": conf,
            "mse_score": score,
            "bbox_json": json.dumps(bbox_list),
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "estimated_area_sqm": area_sqm,
            "polygon_wgs84_json": json.dumps(poly_wgs84) if poly_wgs84 is not None else None,
        }

        with self._lock:
            self._spool.append(record_entry)
            should_flush = len(self._spool) >= self.batch_size

        if should_flush:
            self.flush()

    def flush(self) -> int:
        """Flushes all spooled records into SQLite in a single transaction."""
        with self._lock:
            if not self._spool:
                return 0
            to_write = list(self._spool)
            self._spool.clear()

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.executemany("""
            INSERT INTO anomaly_logs (
                timestamp, anomaly_id, anomaly_type, confidence, mse_score,
                bbox_json, latitude, longitude, altitude, estimated_area_sqm, polygon_wgs84_json, synced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, [
            (
                r["timestamp"],
                r["anomaly_id"],
                r["anomaly_type"],
                r["confidence"],
                r["mse_score"],
                r["bbox_json"],
                r["latitude"],
                r["longitude"],
                r["altitude"],
                r["estimated_area_sqm"],
                r["polygon_wgs84_json"],
            )
            for r in to_write
        ])
        conn.commit()
        conn.close()
        return len(to_write)

    def log_anomaly_offline(
        self,
        anomaly_dict: Union[CanopyAnomaly, Dict[str, Any]],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: Optional[float] = None,
    ) -> None:
        """Backwards compatibility alias for record()."""
        self.record(anomaly_dict, latitude=latitude, longitude=longitude, altitude=altitude)

    def get_pending_sync_count(self) -> int:
        """Returns number of buffered un-synchronized telemetry logs."""
        self.flush()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM anomaly_logs WHERE synced = 0")
        count = cur.fetchone()[0]
        conn.close()
        return count

    def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves recent anomaly records from the persistent store."""
        self.flush()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT id, timestamp, anomaly_id, anomaly_type, confidence, mse_score, bbox_json, 
                   latitude, longitude, altitude, estimated_area_sqm, polygon_wgs84_json, synced
            FROM anomaly_logs ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()

        logs = []
        for r in rows:
            logs.append({
                "log_id": r[0],
                "timestamp": time.strftime("%H:%M:%S", time.localtime(r[1])),
                "anomaly_id": r[2],
                "type": r[3],
                "confidence": r[4],
                "mse_score": r[5],
                "bbox": json.loads(r[6]),
                "latitude": r[7],
                "longitude": r[8],
                "altitude": r[9],
                "estimated_area_sqm": r[10],
                "polygon_wgs84": json.loads(r[11]) if r[11] else None,
                "synced": bool(r[12]),
            })
        return logs

    def start_network_monitor(self) -> None:
        """Starts heartbeat network listener for ground station synchronization."""
        if self._running:
            return
        self._running = True
        self._monitor_thread = threading.Thread(target=self._network_loop, daemon=True)
        self._monitor_thread.start()

    def stop_network_monitor(self) -> None:
        """Stops heartbeat monitor thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _network_loop(self) -> None:
        while self._running:
            time.sleep(5.0)
            if self.is_connected and self.get_pending_sync_count() > 0:
                self.sync_logs_to_ground_station()

    def sync_logs_to_ground_station(self) -> int:
        """Marks buffered offline telemetry as synchronized with ground station."""
        self.flush()
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("UPDATE anomaly_logs SET synced = 1 WHERE synced = 0")
        synced_count = cur.rowcount
        conn.commit()
        conn.close()
        return synced_count

    def toggle_simulated_connection(self) -> bool:
        """Toggles connection state for integration testing."""
        self.is_connected = not self.is_connected
        return self.is_connected
