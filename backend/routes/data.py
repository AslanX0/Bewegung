# Routen fuer Sensordaten (/api/data/*)

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import pymysql

from database import get_db_connection

router = APIRouter()


@router.get("/api/data/latest")
def api_latest():
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500,
                            content={"success": False, "error": "Datenbankverbindung fehlgeschlagen"})
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1")
        data = cursor.fetchone()
        if data and data.get("timestamp"):
            data["timestamp"] = data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return {"success": True, "data": data or {}}
    except pymysql.Error as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        conn.close()


@router.get("/api/data/stats")
def api_stats():
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500,
                            content={"success": False, "error": "Datenbankverbindung fehlgeschlagen"})
    try:
        cursor = conn.cursor()
        time_24h_ago = datetime.now() - timedelta(days=60)
        cursor.execute("""
            SELECT
                COUNT(*) as total_readings,
                AVG(temperature) as avg_temp,
                MAX(temperature) as max_temp,
                MIN(temperature) as min_temp,
                AVG(humidity) as avg_humidity,
                AVG(pressure) as avg_pressure,
                SUM(CASE WHEN movement_detected = 1 THEN 1 ELSE 0 END) as movement_count,
                AVG(estimated_occupancy) as avg_occupancy,
                MAX(estimated_occupancy) as max_occupancy,
                MIN(estimated_occupancy) as min_occupancy
            FROM sensor_data
            WHERE timestamp >= %s
        """, (time_24h_ago,))
        stats = cursor.fetchone()
        return {"success": True, "data": stats or {}}
    except pymysql.Error as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        conn.close()


@router.get("/api/data/history")
def api_history(hours: int = Query(default=1440), limit: int = Query(default=100000)):
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500,
                            content={"success": False, "error": "Datenbankverbindung fehlgeschlagen"})
    try:
        time_ago = datetime.now() - timedelta(hours=hours)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM sensor_data
            WHERE timestamp >= %s
            ORDER BY timestamp ASC
            LIMIT %s
        """, (time_ago, limit))
        data = cursor.fetchall()
        for row in data:
            if row.get("timestamp"):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return {"success": True, "data": data}
    except pymysql.Error as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        conn.close()


@router.get("/api/data/table")
def api_table(page: int = Query(default=1), per_page: int = Query(default=20)):
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500,
                            content={"success": False, "error": "Datenbankverbindung fehlgeschlagen"})
    try:
        offset = (page - 1) * per_page
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM sensor_data")
        total = cursor.fetchone()['total']
        cursor.execute("""
            SELECT * FROM sensor_data
            ORDER BY id DESC
            LIMIT %s OFFSET %s
        """, (per_page, offset))
        data = cursor.fetchall()
        for row in data:
            if row.get("timestamp"):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        pagination = {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }
        return {"success": True, "data": data, "pagination": pagination}
    except pymysql.Error as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        conn.close()


@router.get("/api/data")
def api_data_legacy():
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500, content={"error": "Datenbankverbindung fehlgeschlagen"})
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 10000")
        data = cursor.fetchall()
        for row in data:
            if row.get("timestamp"):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return data
    except pymysql.Error as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        conn.close()
