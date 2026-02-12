# Personenschaetzung (VOC-Baseline) + Lineare Regression (Personen -> Temperatur)
# BME680 (Temperatur, Luftfeuchtigkeit, VOC) + RCWL-0516 (Bewegung/Plausibilitaet)

import numpy as np
import json
import os
from datetime import datetime

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")

MAX_PERSONS = 120
MIN_PERSONS = 0

DEFAULT_BASELINE = {
    "gas_resistance": 200000,
    "calibrated": False,
    "calibration_date": None
}


class PersonEstimator:
    """Schaetzt Personenanzahl aus VOC (gas_resistance) mit Baseline-Kalibrierung.
    Bewegungssensor (RCWL-0516) dient als Plausibilitaetspruefung."""

    def __init__(self):
        self.baseline = DEFAULT_BASELINE.copy()
        self._load_calibration()

    def _load_calibration(self):
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, "r") as f:
                    data = json.load(f)
                    self.baseline = data.get("baseline", DEFAULT_BASELINE.copy())
                    print(f"VOC-Baseline geladen: {self.baseline.get('gas_resistance')} Ohm")
            except Exception as e:
                print(f"Kalibrierungsdatei fehlerhaft: {e}")

    def _save_calibration(self):
        data = {"baseline": self.baseline}
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def set_baseline(self, gas_resistance):
        """VOC-Baseline bei leerem Raum setzen."""
        self.baseline = {
            "gas_resistance": gas_resistance,
            "calibrated": True,
            "calibration_date": datetime.now().isoformat()
        }
        self._save_calibration()
        print(f"VOC-Baseline gesetzt: {gas_resistance} Ohm")

    def estimate(self, gas_resistance, movement_detected=False):
        """Schaetzt Personenanzahl aus VOC-Wert mit Radar-Plausibilitaet.

        Formel: Wenn Gas sinkt gegenueber Baseline -> mehr Personen.
        gas_ratio = aktuell / baseline (< 1.0 = mehr VOC = mehr Personen)
        Exponentielles Modell: persons = -ln(gas_ratio) / k
        wobei k = ln(2) / 60 (bei 60 Personen halbiert sich der Gaswiderstand)
        """
        if not gas_resistance or not self.baseline.get("gas_resistance"):
            return {"estimated_persons": 0, "gas_ratio": None,
                    "movement_plausible": movement_detected}

        baseline_gas = self.baseline["gas_resistance"]
        gas_ratio = gas_resistance / baseline_gas

        if gas_ratio < 1.0:
            k = np.log(2) / 60
            raw_persons = -np.log(gas_ratio) / k
        else:
            raw_persons = 0

        # Plausibilitaetspruefung: Wenn kein Radar-Signal und VOC sagt > 5 Personen
        # -> reduziere Schaetzung (koennte Stoerquelle sein)
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
        return {
            "baseline": self.baseline,
            "max_persons": MAX_PERSONS
        }


class TemperatureRegression:
    """Lineare Regression: Personenanzahl -> Raumtemperatur.
    y = a * x + b  (a = Steigung, b = Achsenabschnitt)
    Trainiert mit Daten der letzten 48 Stunden."""

    def __init__(self):
        self.slope = None         # a (Steigung)
        self.intercept = None     # b (Achsenabschnitt)
        self.r_squared = None     # R²
        self.n_samples = 0
        self.trained_at = None

    def train(self, persons_list, temperature_list):
        """Trainiert lineare Regression mit Messdaten.
        persons_list: Liste der geschaetzten Personenanzahlen (x)
        temperature_list: Liste der gemessenen Temperaturen (y)
        """
        if len(persons_list) < 3:
            print(f"Mindestens 3 Datenpunkte noetig (aktuell: {len(persons_list)})")
            return False

        x = np.array(persons_list, dtype=float)
        y = np.array(temperature_list, dtype=float)

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x ** 2)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            print("Regression nicht moeglich: Alle x-Werte identisch")
            return False

        self.slope = float((n * sum_xy - sum_x * sum_y) / denominator)
        self.intercept = float((sum_y - self.slope * sum_x) / n)

        # R² berechnen
        y_pred = self.slope * x + self.intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        self.n_samples = n
        self.trained_at = datetime.now().isoformat()

        print(f"Regression trainiert: y = {self.slope:.4f}*x + {self.intercept:.2f} "
              f"(R² = {self.r_squared:.4f}, n = {n})")
        return True

    def predict(self, persons):
        """Sagt Temperatur fuer gegebene Personenanzahl vorher."""
        if self.slope is None:
            return None
        return round(self.slope * persons + self.intercept, 2)

    def predict_scenarios(self):
        """Vorhersagen fuer 0, 60, 120 Personen."""
        if self.slope is None:
            return None
        return [
            {"persons": 0, "predicted_temp": self.predict(0),
             "label": "Leeres Restaurant"},
            {"persons": 60, "predicted_temp": self.predict(60),
             "label": "Halbe Auslastung"},
            {"persons": 120, "predicted_temp": self.predict(120),
             "label": "Volle Auslastung"}
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
