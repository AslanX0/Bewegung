# Routen fuer Personenschaetzung (/api/occupancy/*)

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import pymysql

from database import get_db_connection
from modules import PersonEstimator

estimator = PersonEstimator()

router = APIRouter()


@router.get("/api/occupancy/current")
def api_occupancy_current():
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500,
                            content={"success": False, "error": "Datenbankverbindung fehlgeschlagen"})
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1")
        latest = cursor.fetchone()

        if not latest:
            return {"success": True, "data": {
                "estimated_persons": 0, "occupancy_percent": 0, "sensors": None
            }}

        # VOC-basierte Personenschaetzung
        if latest.get('estimated_occupancy') is not None:
            persons = latest['estimated_occupancy']
        else:
            result = estimator.estimate(
                gas_resistance=latest.get('gas_resistance'),
                movement_detected=bool(latest.get('movement_detected', False))
            )
            persons = result['estimated_persons']
            try:
                cursor.execute(
                    "UPDATE sensor_data SET estimated_occupancy = %s WHERE id = %s",
                    (persons, latest['id']))
                conn.commit()
            except Exception:
                pass

        return {"success": True, "data": {
            "estimated_persons": persons,
            "occupancy_percent": round(persons / 120 * 100, 1),
            "sensors": {
                "temperature": latest.get('temperature'),
                "humidity": latest.get('humidity'),
                "pressure": latest.get('pressure'),
                "gas_resistance": latest.get('gas_resistance'),
                "movement_detected": bool(latest.get('movement_detected', False))
            },
            "timestamp": latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                         if latest.get("timestamp") else None
        }}
    except pymysql.Error as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        conn.close()


@router.get("/api/occupancy/history")
def api_occupancy_history(hours: int = Query(default=24)):
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500,
                            content={"success": False, "error": "Datenbankverbindung fehlgeschlagen"})
    try:
        time_ago = datetime.now() - timedelta(hours=hours)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, estimated_occupancy, temperature, humidity,
                   gas_resistance, movement_detected
            FROM sensor_data
            WHERE timestamp >= %s
            ORDER BY timestamp ASC
            LIMIT 500
        """, (time_ago,))
        data = cursor.fetchall()
        for row in data:
            if row.get("timestamp"):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            if row.get("estimated_occupancy") is None:
                est = estimator.estimate(
                    gas_resistance=row.get('gas_resistance'),
                    movement_detected=bool(row.get('movement_detected', False))
                )
                row['estimated_occupancy'] = est['estimated_persons']
        return {"success": True, "data": data}
    except pymysql.Error as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        conn.close()
