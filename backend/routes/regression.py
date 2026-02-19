# Routen fuer Lineare Regression (/api/regression/*)
# Personen -> Temperatur Vorhersage

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import pymysql

from database import get_db_connection
from modules import TemperatureRegression

regression = TemperatureRegression()

router = APIRouter()


def train_regression_from_db(hours=0):
    """Trainiert das Regressionsmodell. hours=0 bedeutet alle verfuegbaren Daten."""
    conn = get_db_connection()
    if conn is None:
        regression.last_error = "Keine Datenbankverbindung"
        return False
    try:
        cursor = conn.cursor()

        if hours > 0:
            time_ago = datetime.now() - timedelta(hours=hours)
            cursor.execute("""
                SELECT estimated_occupancy, temperature
                FROM sensor_data
                WHERE timestamp >= %s
                  AND temperature IS NOT NULL
                  AND estimated_occupancy IS NOT NULL
                ORDER BY timestamp ASC
            """, (time_ago,))
        else:
            cursor.execute("""
                SELECT estimated_occupancy, temperature
                FROM sensor_data
                WHERE temperature IS NOT NULL
                  AND estimated_occupancy IS NOT NULL
                ORDER BY timestamp ASC
            """)

        rows = cursor.fetchall()

        if len(rows) < 3:
            regression.last_error = f"Zu wenig Daten ({len(rows)} Datenpunkte, mind. 3 benoetigt)"
            print(f"[Regression] {regression.last_error}")
            return False

        persons_list = [row['estimated_occupancy'] for row in rows]
        temp_list = [row['temperature'] for row in rows]

        success = regression.train(persons_list, temp_list)
        if success:
            regression.last_error = None
            print(f"[Regression] Trainiert: slope={regression.slope:.4f}, "
                  f"intercept={regression.intercept:.2f}, R²={regression.r_squared:.4f}, "
                  f"n={regression.n_samples}")
        else:
            regression.last_error = "Training fehlgeschlagen (keine Varianz in den Daten?)"
        return success
    except Exception as e:
        regression.last_error = str(e)
        print(f"[Regression] Trainingsfehler: {e}")
        return False
    finally:
        conn.close()


@router.get("/api/regression/status")
def api_regression_status():
    status = regression.get_status()
    status["last_error"] = getattr(regression, 'last_error', None)
    return {"success": True, "data": status}


@router.get("/api/regression/predict")
def api_regression_predict(persons: int = Query(default=60, ge=0, le=120)):
    temp = regression.predict(persons)
    if temp is None:
        return JSONResponse(status_code=400,
                            content={"success": False, "error": "Modell noch nicht trainiert"})
    return {"success": True, "data": {
        "persons": persons, "predicted_temperature": temp
    }}


@router.get("/api/regression/scatter")
def api_regression_scatter(hours: int = Query(default=0)):
    """Scatter-Daten: Personenanzahl (x) vs Temperatur (y) fuer Diagramm.
    hours=0 liefert alle verfuegbaren Daten."""
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500,
                            content={"success": False, "error": "Datenbankverbindung fehlgeschlagen"})
    try:
        cursor = conn.cursor()

        if hours > 0:
            time_ago = datetime.now() - timedelta(hours=hours)
            cursor.execute("""
                SELECT estimated_occupancy, temperature
                FROM sensor_data
                WHERE timestamp >= %s
                  AND temperature IS NOT NULL
                  AND estimated_occupancy IS NOT NULL
                ORDER BY timestamp ASC
            """, (time_ago,))
        else:
            cursor.execute("""
                SELECT estimated_occupancy, temperature
                FROM sensor_data
                WHERE temperature IS NOT NULL
                  AND estimated_occupancy IS NOT NULL
                ORDER BY timestamp ASC
            """)

        rows = cursor.fetchall()
        points = [{"x": row['estimated_occupancy'], "y": row['temperature']} for row in rows]

        regression_line = None
        if regression.slope is not None and points:
            x_min = min(p['x'] for p in points)
            x_max = max(p['x'] for p in points)
            line_start = max(0, x_min - 5)
            line_end = min(130, x_max + 5)
            regression_line = {
                "slope": regression.slope,
                "intercept": regression.intercept,
                "r_squared": regression.r_squared,
                "points": [
                    {"x": line_start, "y": regression.predict(line_start)},
                    {"x": line_end,   "y": regression.predict(line_end)}
                ]
            }

        return {"success": True, "data": {
            "points": points,
            "count": len(points),
            "regression_line": regression_line,
            "scenarios": regression.predict_scenarios()
        }}
    except pymysql.Error as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        conn.close()


@router.post("/api/regression/train")
def api_regression_train(hours: int = Query(default=0)):
    """Manuelles Neutrainieren des Modells. hours=0 nutzt alle Daten."""
    success = train_regression_from_db(hours=hours)
    if success:
        return {"success": True, "message": "Modell trainiert",
                "data": regression.get_status()}
    return JSONResponse(status_code=400,
                        content={"success": False,
                                 "error": getattr(regression, 'last_error', 'Unbekannter Fehler')})
