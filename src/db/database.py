import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional
from src.models.user import UserModel
from src.models.scan import ThermalScanModel

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DB_PATH = os.path.join(DB_DIR, "thermalriskai.db")


def get_db_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        uid TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        photo_url TEXT,
        created_at TEXT NOT NULL,
        last_login TEXT NOT NULL
    )
    """)

    # Thermal Scans Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS thermal_scans (
        scan_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        uploaded_image TEXT,
        prediction_json TEXT,
        ndvii_json TEXT,
        organ_mapping_json TEXT,
        gradcam TEXT,
        wellness_score REAL NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users (uid)
    )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", DB_PATH)


# Initialize DB automatically when module is imported
init_db()


def upsert_user(user_data: dict) -> UserModel:
    conn = get_db_connection()
    cursor = conn.cursor()
    now_iso = datetime.utcnow().isoformat()

    uid = user_data["uid"]
    name = user_data.get("name", "User")
    email = user_data.get("email", "")
    photo_url = user_data.get("picture") or user_data.get("photoURL", "")

    cursor.execute("SELECT created_at FROM users WHERE uid = ?", (uid,))
    row = cursor.fetchone()

    if row:
        created_at = row["created_at"]
        cursor.execute("""
            UPDATE users SET name = ?, email = ?, photo_url = ?, last_login = ?
            WHERE uid = ?
        """, (name, email, photo_url, now_iso, uid))
    else:
        created_at = now_iso
        cursor.execute("""
            INSERT INTO users (uid, name, email, photo_url, created_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (uid, name, email, photo_url, created_at, now_iso))

    conn.commit()
    conn.close()

    return UserModel(
        uid=uid,
        name=name,
        email=email,
        photoURL=photo_url,
        createdAt=created_at,
        lastLogin=now_iso
    )


def get_user(uid: str) -> Optional[UserModel]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE uid = ?", (uid,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return UserModel(
        uid=row["uid"],
        name=row["name"],
        email=row["email"],
        photoURL=row["photo_url"],
        createdAt=row["created_at"],
        lastLogin=row["last_login"]
    )


def save_scan(scan_model: ThermalScanModel) -> ThermalScanModel:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO thermal_scans (
            scan_id, user_id, timestamp, uploaded_image,
            prediction_json, ndvii_json, organ_mapping_json,
            gradcam, wellness_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scan_model.scanId,
        scan_model.userId,
        scan_model.timestamp,
        scan_model.uploadedImage,
        json.dumps(scan_model.prediction),
        json.dumps(scan_model.NDVII),
        json.dumps(scan_model.OrganMapping),
        scan_model.GradCAM,
        scan_model.WellnessScore
    ))

    conn.commit()
    conn.close()
    return scan_model


def get_user_scans(user_id: str, limit: int = 100) -> List[ThermalScanModel]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM thermal_scans
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()

    scans = []
    for r in rows:
        scans.append(ThermalScanModel(
            scanId=r["scan_id"],
            userId=r["user_id"],
            timestamp=r["timestamp"],
            uploadedImage=r["uploaded_image"] or "",
            prediction=json.loads(r["prediction_json"]) if r["prediction_json"] else {},
            NDVII=json.loads(r["ndvii_json"]) if r["ndvii_json"] else {},
            OrganMapping=json.loads(r["organ_mapping_json"]) if r["organ_mapping_json"] else {},
            GradCAM=r["gradcam"] or "",
            WellnessScore=r["wellness_score"]
        ))
    return scans


def get_latest_scan(user_id: str) -> Optional[ThermalScanModel]:
    scans = get_user_scans(user_id, limit=1)
    return scans[0] if scans else None
