# Routen fuer Lineare Regression (/api/regression/*)
# Personen -> Temperatur Vorhersage

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import pymysql

from database import get_db_connection
from modules import TemperatureRegression
from routes.occupancy import estimator

regression = TemperatureRegression()

router = APIRouter()


def train_regression_from_db(hours=48):
    """Trainiert das Regressionsmodell mit Daten der letzten X Stunden."""
    conn = get_db_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        time_ago = datetime.now() - timedelta(hours=hours)
        cursor.execute("""
            SELECT estimated_occupancy, temperature, gas_resistance, movement_detected
            FROM sensor_data
            WHERE timestamp >= %s
              AND temperature IS NOT NULL
            ORDER BY timestamp ASC
        """, (time_ago,))
        rows = cursor.fetchall()

        persons_list = []
        temp_list = []
        for row in rows:
            # Personenzahl aus DB oder live berechnen
            if row.get('estimated_occupancy') is not None:
                persons = row['estimated_occupancy']
            else:
                est = estimator.estimate(
                    gas_resistance=row.get('gas_resistance'),
                    movement_detected=bool(row.get('movement_detected', False))
                )
                persons = est['estimated_persons']
            persons_list.append(persons)
            temp_list.append(row['temperature'])

        if len(persons_list) >= 3:
            return regression.train(persons_list, temp_list)
        return False
    except Exception as e:
        print(f"[Regression] Trainingsfehler: {e}")
        return False
    finally:
        conn.close()


@router.get("/api/regression/status")
def api_regression_status():
    return {"success": True, "data": regression.get_status()}


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
def api_regression_scatter(hours: int = Query(default=48)):
    """Scatter-Daten: Personenanzahl (x) vs Temperatur (y) fuer Diagramm."""
    conn = get_db_connection()
    if conn is None:
        return JSONResponse(status_code=500,
                            content={"success": False, "error": "Datenbankverbindung fehlgeschlagen"})
    try:
        cursor = conn.cursor()
        time_ago = datetime.now() - timedelta(hours=hours)
        cursor.execute("""
            SELECT estimated_occupancy, temperature, gas_resistance, movement_detected
            FROM sensor_data
            WHERE timestamp >= %s
              AND temperature IS NOT NULL
            ORDER BY timestamp ASC
        """, (time_ago,))
        rows = cursor.fetchall()

        points = []
        for row in rows:
            if row.get('estimated_occupancy') is not None:
                persons = row['estimated_occupancy']
            else:
                est = estimator.estimate(
                    gas_resistance=row.get('gas_resistance'),
                    movement_detected=bool(row.get('movement_detected', False))
                )
                persons = est['estimated_persons']
            points.append({"x": persons, "y": row['temperature']})

        # Regressionslinie fuer Chart (2 Punkte genuegen)
        regression_line = None
        if regression.slope is not None:
            regression_line = {
                "slope": regression.slope,
                "intercept": regression.intercept,
                "r_squared": regression.r_squared,
                "points": [
                    {"x": 0, "y": regression.predict(0)},
                    {"x": 120, "y": regression.predict(120)}
                ]
            }

        return {"success": True, "data": {
            "points": points,
            "regression_line": regression_line,
            "scenarios": regression.predict_scenarios()
        }}
    except pymysql.Error as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        conn.close()


@router.post("/api/regression/train")
def api_regression_train():
    """Manuelles Neutrainieren des Modells."""
    success = train_regression_from_db(hours=48)
    if success:
        return {"success": True, "message": "Modell trainiert",
                "data": regression.get_status()}
    return JSONResponse(status_code=400,
                        content={"success": False,
                                 "error": "Training fehlgeschlagen (zu wenig Daten?)"})
