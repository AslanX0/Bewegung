from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from database import get_db_connection
from modules import PersonEstimator

router = APIRouter()
estimator = PersonEstimator()


def _get_persons(row, cursor=None, conn=None):
    if row.get('estimated_occupancy') is not None:
        return row['estimated_occupancy']
    
    persons = estimator.estimate(
        row.get('gas_resistance'),
        bool(row.get('movement_detected', False))
    )['estimated_persons']
    
    if cursor and conn and row.get('id'):
        cursor.execute("UPDATE sensor_data SET estimated_occupancy=%s WHERE id=%s", (persons, row['id']))
        conn.commit()
    
    return persons


@router.get("/api/occupancy/current")
def current():
    conn = get_db_connection()
    if not conn:
        return JSONResponse(status_code=500, content={"success": False, "error": "DB-Fehler"})
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        
        if not row:
            return {"success": True, "data": {"estimated_persons": 0, "occupancy_percent": 0, "sensors": None}}
        
        persons = _get_persons(row, cursor, conn)
        
        return {"success": True, "data": {
            "estimated_persons": persons,
            "occupancy_percent": round(persons / 120 * 100, 1),
            "sensors": {
                "temperature": row.get('temperature'),
                "humidity": row.get('humidity'),
                "pressure": row.get('pressure'),
                "gas_resistance": row.get('gas_resistance'),
                "movement_detected": bool(row.get('movement_detected'))
            },
            "timestamp": row['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if row.get('timestamp') else None
        }}
    finally:
        conn.close()


@router.get("/api/occupancy/history")
def history(hours: int = Query(1440)):
    conn = get_db_connection()
    if not conn:
        return JSONResponse(status_code=500, content={"success": False, "error": "DB-Fehler"})
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, estimated_occupancy, temperature, gas_resistance, movement_detected
            FROM sensor_data WHERE timestamp >= %s ORDER BY timestamp LIMIT 100000
        """, (datetime.now() - timedelta(hours=hours),))
        
        rows = cursor.fetchall()
        for r in rows:
            r['estimated_occupancy'] = _get_persons(r, cursor, conn)
            r['timestamp'] = r['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if r.get('timestamp') else None
        
        return {"success": True, "data": rows}
    finally:
        conn.close()