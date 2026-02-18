# FastAPI-Server - Restaurant Sensor API
# BME680 (Temperatur, Feuchtigkeit, VOC) + RCWL-0516 (Bewegung)
# Personenschaetzung (VOC-Baseline) + Lineare Regression (Personen -> Temperatur)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio

from database import get_db_connection, db_config
import pymysql
from routes import data_router, occupancy_router, estimator_router, regression_router
from routes.occupancy import estimator
from routes.regression import regression, train_regression_from_db


async def estimation_loop():
    while True:
        try:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, gas_resistance, movement_detected
                        FROM sensor_data
                        WHERE estimated_occupancy IS NULL
                        ORDER BY id ASC
                        LIMIT 500
                    """)
                    rows = cursor.fetchall()
                    for row in rows:
                        result = estimator.estimate(
                            gas_resistance=row.get('gas_resistance'),
                            movement_detected=bool(row.get('movement_detected', False))
                        )
                        persons = result['estimated_persons']
                        cursor.execute(
                            "UPDATE sensor_data SET estimated_occupancy = %s WHERE id = %s",
                            (persons, row['id']))
                    conn.commit()
                    if rows:
                        print(f"[AutoEstimator] {len(rows)} Datensaetze geschaetzt")
                finally:
                    conn.close()
        except Exception as e:
            print(f"[AutoEstimator] Fehler: {e}")
        await asyncio.sleep(60)


async def regression_train_loop():
    """Trainiert Regressionsmodell alle 6 Stunden mit allen verfuegbaren Daten."""
    # Erstes Training direkt beim Start
    await asyncio.sleep(10)
    train_regression_from_db(hours=0)
    print("[Regression] Initiales Training abgeschlossen")

    while True:
        await asyncio.sleep(6 * 3600)  # Alle 6 Stunden neu trainieren
        try:
            train_regression_from_db(hours=0)
            print("[Regression] Modell neu trainiert (6h-Zyklus)")
        except Exception as e:
            print(f"[Regression] Trainingsfehler: {e}")


def init_db():
    conn = get_db_connection()
    if conn is None:
        print("[DB] WARNUNG: Keine Datenbankverbindung")
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS sensor_data ("
            "id INT AUTO_INCREMENT PRIMARY KEY, "
            "timestamp DATETIME NOT NULL, "
            "temperature FLOAT NOT NULL, "
            "pressure FLOAT, "
            "humidity FLOAT, "
            "gas_resistance FLOAT, "
            "movement_detected BOOLEAN NOT NULL, "
            "estimated_occupancy INT DEFAULT NULL, "
            "ac_recommendation INT DEFAULT NULL)"
        )
        conn.commit()
        print("[DB] Tabelle sensor_data bereit")
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(app):
    init_db()
    task_est = asyncio.create_task(estimation_loop())
    task_reg = asyncio.create_task(regression_train_loop())
    print("[Server] Hintergrund-Tasks gestartet")
    print("  - AutoEstimator: Personenschaetzung alle 60s")
    print("  - Regression: Modelltraining alle 48h")
    yield
    task_est.cancel()
    task_reg.cancel()


app = FastAPI(title="Asia Restaurant Sensor API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"], allow_credentials=True)

app.include_router(data_router)
app.include_router(occupancy_router)
app.include_router(estimator_router)
app.include_router(regression_router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("   ASIA RESTAURANT - Sensor Dashboard")
    print("   " + "=" * 56)
    print("   Dashboard:      http://localhost:8000/")
    print("   API Docs:       http://localhost:8000/docs")
    print("   Regression:     http://localhost:8000/api/regression/status")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
