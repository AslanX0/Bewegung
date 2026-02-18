import numpy as np
import json
import os
from datetime import datetime

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")
REGRESSION_FILE = os.path.join(os.path.dirname(__file__), "regression_model.json")

MAX_PERSONS = 120
MIN_PERSONS = 0

DEFAULT_BASELINE = {
    "gas_resistance": 200000,
    "calibrated": False,
    "calibration_date": None
}


class PersonEstimator:
    # Schätzt Personenanzahl anhand des VOC-Werts (BME680 gas_resistance)

    def __init__(self):
        self.baseline = DEFAULT_BASELINE.copy()
        self._load_calibration()

    def _load_calibration(self):
        if not os.path.exists(CALIBRATION_FILE):
            return
        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
                self.baseline = data.get("baseline", DEFAULT_BASELINE.copy())
        except Exception as e:
            print(f"Fehler beim Laden der Kalibrierung: {e}")

    def _save_calibration(self):
        with open(CALIBRATION_FILE, "w") as f:
            json.dump({"baseline": self.baseline}, f, indent=2, default=str)

    def set_baseline(self, gas_resistance):
        """Setzt die VOC-Baseline (bei leerem Raum aufrufen)."""
        self.baseline = {
            "gas_resistance": gas_resistance,
            "calibrated": True,
            "calibration_date": datetime.now().isoformat()
        }
        self._save_calibration()

    def estimate(self, gas_resistance, movement_detected=False):
        """
        Berechnet Personenschätzung aus aktuellem Gaswiderstand.
        Prinzip: Mehr Personen -> mehr VOC -> niedrigerer Gaswiderstand
        """
        if not gas_resistance or not self.baseline.get("gas_resistance"):
            return {
                "estimated_persons": 0,
                "gas_ratio": None,
                "movement_plausible": movement_detected
            }

        baseline_gas = self.baseline["gas_resistance"]
        gas_ratio = gas_resistance / baseline_gas

        # Exponentielles Modell: persons = -ln(ratio) / k
        if gas_ratio < 1.0:
            k = np.log(2) / 60
            raw_persons = -np.log(gas_ratio) / k
        else:
            raw_persons = 0

        # Ohne Bewegung aber hohe VOC-Werte -> wahrscheinlich Störquelle
        if not movement_detected and raw_persons > 5:
            raw_persons *= 0.3

        estimated = int(np.clip(round(raw_persons), MIN_PERSONS, MAX_PERSONS))

        return {
            "estimated_persons": estimated,
            "gas_ratio": round(gas_ratio, 4),
            "movement_detected": movement_detected,
            "movement_plausible": movement_detected or estimated <= 5,
            "baseline_calibrated": self.baseline.get("calibrated", False)
        }

    def get_status(self):
        return {"baseline": self.baseline, "max_persons": MAX_PERSONS}


class TemperatureRegression:
    """
    Lineare Regression: Personenanzahl -> Raumtemperatur
    Wird mit Daten der letzten 48h trainiert.
    """

    def __init__(self):
        self.slope = None
        self.intercept = None
        self.r_squared = None
        self.n_samples = 0
        self.trained_at = None
        self.last_error = None
        self._load_model()

    def train(self, persons_list, temperature_list):
        """Trainiert das Modell mit gesammelten Messdaten."""
        if len(persons_list) < 3:
            return False

        x = np.array(persons_list, dtype=float)
        y = np.array(temperature_list, dtype=float)
        n = len(x)

        sum_x, sum_y = np.sum(x), np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            return False

        self.slope = float((n * sum_xy - sum_x * sum_y) / denom)
        self.intercept = float((sum_y - self.slope * sum_x) / n)

        y_pred = self.slope * x + self.intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        self.n_samples = n
        self.trained_at = datetime.now().isoformat()
        self._save_model()
        return True

    def _save_model(self):
        try:
            with open(REGRESSION_FILE, "w") as f:
                json.dump({
                    "slope": self.slope,
                    "intercept": self.intercept,
                    "r_squared": self.r_squared,
                    "n_samples": self.n_samples,
                    "trained_at": self.trained_at
                }, f, indent=2)
        except Exception as e:
            print(f"[Regression] Fehler beim Speichern: {e}")

    def _load_model(self):
        if not os.path.exists(REGRESSION_FILE):
            return
        try:
            with open(REGRESSION_FILE, "r") as f:
                data = json.load(f)
            self.slope = data.get("slope")
            self.intercept = data.get("intercept")
            self.r_squared = data.get("r_squared")
            self.n_samples = data.get("n_samples", 0)
            self.trained_at = data.get("trained_at")
            if self.slope is not None:
                print(f"[Regression] Modell geladen (R²={self.r_squared}, n={self.n_samples})")
        except Exception as e:
            print(f"[Regression] Fehler beim Laden: {e}")

    def predict(self, persons):
        """Gibt vorhergesagte Temperatur für Personenanzahl zurück."""
        if self.slope is None:
            return None
        return round(self.slope * persons + self.intercept, 2)

    def predict_scenarios(self):
        """Standardszenarien: leer, halb voll, voll."""
        if self.slope is None:
            return None
        return [
            {"persons": 0, "predicted_temp": self.predict(0), "label": "Leer"},
            {"persons": 60, "predicted_temp": self.predict(60), "label": "Halb"},
            {"persons": 120, "predicted_temp": self.predict(120), "label": "Voll"}
        ]

    def get_status(self):
        return {
            "trained": self.slope is not None,
            "slope": round(self.slope, 6) if self.slope is not None else None,
            "intercept": round(self.intercept, 4) if self.intercept is not None else None,
            "r_squared": round(self.r_squared, 4) if self.r_squared is not None else None,
            "n_samples": self.n_samples,
            "trained_at": self.trained_at,
            "scenarios": self.predict_scenarios()
        }