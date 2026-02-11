"""
===============================================================================
 PERSONENSCHÄTZUNG – Regressionsanalyse
 Schätzt die Anzahl der Personen im Restaurant (0–120) anhand von Sensordaten
 des BME680 (Temperatur, Feuchtigkeit, Gaswiderstand) und PIR-Bewegungssensor.
===============================================================================

 PHYSIKALISCHES MODELL:
 ─────────────────────
 Jede Person erzeugt messbare Umweltveränderungen:
   • Wärmeabgabe:    ~80–120W → Temperaturanstieg
   • Atemluft:       ~40 g/h Wasserdampf → Feuchtigkeitsanstieg
   • CO2 / Gerüche:  → Gaswiderstand sinkt (VOC steigen)
   • Bewegung:       → PIR-Aktivität steigt

 REGRESSION:
 ───────────
 Das Modell nutzt Multiple Lineare Regression auf die Delta-Werte
 (Abweichung von der Baseline bei leerem Restaurant).
 
 Geschätzte Personen = β₀ + β₁·ΔTemp + β₂·ΔHumidity + β₃·ΔGas + β₄·Motion

 KALIBRIERUNG:
 ─────────────
 Phase 1: Physikalisches Modell (sofort nutzbar, Schätzwerte)
 Phase 2: Datengetrieben (nachdem manuell Personenzahlen erfasst wurden)
===============================================================================
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta

# ==============================================================================
# KONFIGURATION
# ==============================================================================

# Pfad zur Kalibrierungsdatei
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")

# Restaurant-Parameter
MAX_PERSONS = 120
MIN_PERSONS = 0

# Baseline-Werte (leeres Restaurant, Klimaanlage Stufe 3)
# Diese Werte MÜSSEN bei leerem Restaurant kalibriert werden!
DEFAULT_BASELINE = {
    "temperature": 22.0,      # °C bei leerem Restaurant
    "humidity": 40.0,          # %RH bei leerem Restaurant
    "gas_resistance": 200000,  # Ohm bei sauberer Luft (leer)
    "calibrated": False,
    "calibration_date": None
}

# Physikalisches Modell – Koeffizienten (Phase 1)
# Basierend auf typischen Werten für einen ~150m² Gastraum
PHYSICAL_MODEL = {
    # Pro Person: ca. +0.05°C in einem 150m² Raum
    "temp_per_person": 0.05,
    # Pro Person: ca. +0.15% RH
    "humidity_per_person": 0.15,
    # Gaswiderstand halbiert sich bei ca. 60 Personen (logarithmisch)
    "gas_half_persons": 60,
    # Bewegungsaktivität: Gewichtungsfaktor
    "motion_weight": 5.0
}


# ==============================================================================
# KLASSE: PersonEstimator
# ==============================================================================

class PersonEstimator:
    """
    Schätzt die Personenanzahl im Restaurant anhand der Sensordaten.
    
    Zwei Modi:
    1. Physikalisches Modell (Standard): Nutzt bekannte physikalische
       Zusammenhänge zwischen Personen und Umweltveränderungen.
    2. Trainiertes Modell: Nutzt gespeicherte Kalibrierungsdaten
       (nachdem manuell Personenzahlen erfasst wurden).
    """

    def __init__(self):
        self.baseline = DEFAULT_BASELINE.copy()
        self.trained_coefficients = None
        self.training_data = []
        self._load_calibration()

    # ──────────────────────────────────────────────────────────────────────
    # Kalibrierung laden / speichern
    # ──────────────────────────────────────────────────────────────────────

    def _load_calibration(self):
        """Lädt gespeicherte Kalibrierungsdaten."""
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, "r") as f:
                    data = json.load(f)
                    self.baseline = data.get("baseline", DEFAULT_BASELINE.copy())
                    self.trained_coefficients = data.get("coefficients", None)
                    self.training_data = data.get("training_data", [])
                    print(f"✓ Kalibrierung geladen ({len(self.training_data)} Trainingspunkte)")
            except Exception as e:
                print(f"⚠ Kalibrierungsdatei fehlerhaft: {e}")

    def _save_calibration(self):
        """Speichert Kalibrierungsdaten."""
        data = {
            "baseline": self.baseline,
            "coefficients": self.trained_coefficients,
            "training_data": self.training_data
        }
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # ──────────────────────────────────────────────────────────────────────
    # Baseline setzen (leeres Restaurant)
    # ──────────────────────────────────────────────────────────────────────

    def set_baseline(self, temperature, humidity, gas_resistance):
        """
        Setzt die Baseline-Werte für ein leeres Restaurant.
        MUSS einmal ausgeführt werden, wenn das Restaurant leer ist!
        
        Args:
            temperature:    Temperatur in °C bei leerem Restaurant
            humidity:       Luftfeuchtigkeit in %RH bei leerem Restaurant
            gas_resistance: Gaswiderstand in Ohm bei leerem Restaurant
        """
        self.baseline = {
            "temperature": temperature,
            "humidity": humidity,
            "gas_resistance": gas_resistance,
            "calibrated": True,
            "calibration_date": datetime.now().isoformat()
        }
        self._save_calibration()
        print(f"✓ Baseline gesetzt: {temperature}°C / {humidity}%RH / {gas_resistance}Ω")

    # ──────────────────────────────────────────────────────────────────────
    # Schätzung: Physikalisches Modell (Phase 1)
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_physical(self, temperature, humidity, gas_resistance, 
                           movement_detected, movement_rate=None):
        """
        Schätzt Personen anhand des physikalischen Modells.
        
        Einzelschätzer:
        1. Temperatur-Delta → Personen
        2. Feuchtigkeits-Delta → Personen
        3. Gaswiderstand-Ratio → Personen (logarithmisch)
        4. Bewegungsrate → Personen
        
        Gewichteter Durchschnitt der Einzelschätzer.
        """
        estimates = {}
        weights = {}

        # --- Temperaturbasierte Schätzung ---
        delta_temp = temperature - self.baseline["temperature"]
        if delta_temp > 0:
            est_temp = delta_temp / PHYSICAL_MODEL["temp_per_person"]
            estimates["temperature"] = est_temp
            weights["temperature"] = 0.25
        else:
            estimates["temperature"] = 0
            weights["temperature"] = 0.10

        # --- Feuchtigkeitsbasierte Schätzung ---
        delta_humidity = humidity - self.baseline["humidity"]
        if delta_humidity > 0:
            est_humidity = delta_humidity / PHYSICAL_MODEL["humidity_per_person"]
            estimates["humidity"] = est_humidity
            weights["humidity"] = 0.30
        else:
            estimates["humidity"] = 0
            weights["humidity"] = 0.10

        # --- Gasbasierte Schätzung (logarithmisch) ---
        if gas_resistance and self.baseline["gas_resistance"]:
            gas_ratio = gas_resistance / self.baseline["gas_resistance"]
            if gas_ratio < 1.0:
                # Logarithmische Beziehung: Widerstand sinkt mit Personen
                # Bei gas_half_persons Personen ist der Widerstand halbiert
                k = np.log(2) / PHYSICAL_MODEL["gas_half_persons"]
                est_gas = -np.log(gas_ratio) / k
                estimates["gas"] = max(0, est_gas)
                weights["gas"] = 0.35
            else:
                estimates["gas"] = 0
                weights["gas"] = 0.10
        else:
            estimates["gas"] = 0
            weights["gas"] = 0.0

        # --- Bewegungsbasierte Schätzung ---
        if movement_rate is not None:
            # movement_rate = Anteil der Messungen mit Bewegung in letzten 30 Min.
            est_motion = movement_rate * MAX_PERSONS * 0.8
            estimates["motion"] = est_motion
            weights["motion"] = 0.10
        elif movement_detected:
            estimates["motion"] = PHYSICAL_MODEL["motion_weight"]
            weights["motion"] = 0.05
        else:
            estimates["motion"] = 0
            weights["motion"] = 0.05

        # --- Gewichteter Durchschnitt ---
        total_weight = sum(weights.values())
        if total_weight > 0:
            weighted_sum = sum(estimates[k] * weights[k] for k in estimates)
            raw_estimate = weighted_sum / total_weight
        else:
            raw_estimate = 0

        # Begrenzen auf 0–120
        final_estimate = int(np.clip(round(raw_estimate), MIN_PERSONS, MAX_PERSONS))

        return {
            "estimated_persons": final_estimate,
            "confidence": self._calculate_confidence(estimates, weights),
            "model": "physical",
            "details": {
                "delta_temperature": round(delta_temp, 2),
                "delta_humidity": round(delta_humidity, 2),
                "gas_ratio": round(gas_resistance / self.baseline["gas_resistance"], 3) 
                             if gas_resistance and self.baseline["gas_resistance"] else None,
                "individual_estimates": {k: round(v, 1) for k, v in estimates.items()},
                "weights": weights
            },
            "baseline_calibrated": self.baseline["calibrated"],
            "climate_recommendation": self._climate_recommendation(final_estimate)
        }

    def _calculate_confidence(self, estimates, weights):
        """
        Berechnet einen Konfidenzwert (0–100%).
        Hohe Konfidenz wenn alle Schätzer ähnliche Werte liefern.
        """
        values = [v for v in estimates.values() if v > 0]
        if len(values) < 2:
            return 30  # Wenig Daten → niedrige Konfidenz

        # Variationskoeffizient
        mean = np.mean(values)
        if mean == 0:
            return 50
        cv = np.std(values) / mean

        # Konfidenz: niedrig bei hoher Variation
        confidence = max(20, min(95, int(100 - cv * 50)))

        # Bonus wenn Baseline kalibriert
        if self.baseline["calibrated"]:
            confidence = min(95, confidence + 10)

        # Bonus wenn trainiertes Modell verfügbar
        if self.trained_coefficients:
            confidence = min(95, confidence + 15)

        return confidence

    # ──────────────────────────────────────────────────────────────────────
    # Schätzung: Trainiertes Modell (Phase 2)
    # ──────────────────────────────────────────────────────────────────────

    def _estimate_trained(self, temperature, humidity, gas_resistance, 
                          movement_detected, movement_rate=None):
        """
        Schätzt Personen anhand des trainierten linearen Regressionsmodells.
        Benötigt mindestens 10 Trainingsdatenpunkte.
        """
        if not self.trained_coefficients:
            return self._estimate_physical(
                temperature, humidity, gas_resistance, 
                movement_detected, movement_rate
            )

        coeff = self.trained_coefficients
        
        # Feature-Vektor aufbauen
        delta_temp = temperature - self.baseline["temperature"]
        delta_humidity = humidity - self.baseline["humidity"]
        gas_ratio = (gas_resistance / self.baseline["gas_resistance"]
                     if gas_resistance and self.baseline["gas_resistance"]
                     else 1.0)
        motion_val = float(movement_detected) if movement_rate is None else movement_rate

        # Lineare Regression: y = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + β₄x₄
        raw_estimate = (
            coeff["intercept"]
            + coeff["beta_temp"] * delta_temp
            + coeff["beta_humidity"] * delta_humidity
            + coeff["beta_gas"] * gas_ratio
            + coeff["beta_motion"] * motion_val
        )

        final_estimate = int(np.clip(round(raw_estimate), MIN_PERSONS, MAX_PERSONS))

        return {
            "estimated_persons": final_estimate,
            "confidence": min(95, 60 + len(self.training_data)),
            "model": "trained_regression",
            "details": {
                "delta_temperature": round(delta_temp, 2),
                "delta_humidity": round(delta_humidity, 2),
                "gas_ratio": round(gas_ratio, 3),
                "coefficients": coeff,
                "training_samples": len(self.training_data)
            },
            "baseline_calibrated": self.baseline["calibrated"],
            "climate_recommendation": self._climate_recommendation(final_estimate)
        }

    # ──────────────────────────────────────────────────────────────────────
    # Hauptmethode: Schätzung
    # ──────────────────────────────────────────────────────────────────────

    def estimate(self, temperature, humidity, gas_resistance=None, 
                 movement_detected=False, movement_rate=None):
        """
        Schätzt die Personenanzahl im Restaurant.
        
        Args:
            temperature:      Aktuelle Temperatur in °C
            humidity:         Aktuelle Luftfeuchtigkeit in %RH
            gas_resistance:   Aktueller Gaswiderstand in Ohm (optional)
            movement_detected: Aktuelle PIR-Erkennung (True/False)
            movement_rate:    Anteil Bewegungs-Positiv in den letzten 30 Min.
                              (0.0–1.0, optional)
        
        Returns:
            dict mit estimated_persons, confidence, model, details, 
            climate_recommendation
        """
        # Trainiertes Modell bevorzugen wenn verfügbar
        if self.trained_coefficients and len(self.training_data) >= 10:
            return self._estimate_trained(
                temperature, humidity, gas_resistance,
                movement_detected, movement_rate
            )
        
        return self._estimate_physical(
            temperature, humidity, gas_resistance,
            movement_detected, movement_rate
        )

    # ──────────────────────────────────────────────────────────────────────
    # Training: Manuell Personenzahl erfassen
    # ──────────────────────────────────────────────────────────────────────

    def add_training_point(self, actual_persons, temperature, humidity, 
                           gas_resistance=None, movement_detected=False):
        """
        Fügt einen Trainingsdatenpunkt hinzu (manuell gezählte Personen).
        
        Args:
            actual_persons: Tatsächlich gezählte Personen (0–120)
            temperature:    Temperatur zum Zeitpunkt der Zählung
            humidity:       Feuchtigkeit zum Zeitpunkt der Zählung
            gas_resistance: Gaswiderstand zum Zeitpunkt der Zählung
            movement_detected: PIR-Status zum Zeitpunkt der Zählung
        """
        if not (MIN_PERSONS <= actual_persons <= MAX_PERSONS):
            raise ValueError(f"Personenzahl muss zwischen {MIN_PERSONS} und {MAX_PERSONS} liegen")

        point = {
            "timestamp": datetime.now().isoformat(),
            "actual_persons": actual_persons,
            "temperature": temperature,
            "humidity": humidity,
            "gas_resistance": gas_resistance,
            "movement_detected": movement_detected
        }
        self.training_data.append(point)
        self._save_calibration()

        # Automatisch neu trainieren wenn genug Daten vorhanden
        if len(self.training_data) >= 10:
            self.train()

        print(f"✓ Trainingspunkt hinzugefügt ({len(self.training_data)} gesamt)")

    def train(self):
        """
        Trainiert das lineare Regressionsmodell mit den gesammelten Daten.
        Benötigt mindestens 10 Datenpunkte.
        
        Methode: Ordinary Least Squares (OLS)
        y = β₀ + β₁·ΔTemp + β₂·ΔHumidity + β₃·GasRatio + β₄·Motion
        """
        if len(self.training_data) < 10:
            print(f"⚠ Mindestens 10 Trainingspunkte nötig (aktuell: {len(self.training_data)})")
            return None

        # Feature-Matrix aufbauen
        X = []
        y = []

        for point in self.training_data:
            delta_temp = point["temperature"] - self.baseline["temperature"]
            delta_humidity = point["humidity"] - self.baseline["humidity"]
            gas_ratio = (point["gas_resistance"] / self.baseline["gas_resistance"]
                         if point.get("gas_resistance") and self.baseline["gas_resistance"]
                         else 1.0)
            motion = float(point.get("movement_detected", False))

            X.append([1.0, delta_temp, delta_humidity, gas_ratio, motion])
            y.append(point["actual_persons"])

        X = np.array(X)
        y = np.array(y)

        # OLS: β = (X^T X)^(-1) X^T y
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ y
        except np.linalg.LinAlgError:
            print("⚠ Matrix nicht invertierbar – Pseudoinverse wird verwendet")
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

        self.trained_coefficients = {
            "intercept": round(float(beta[0]), 4),
            "beta_temp": round(float(beta[1]), 4),
            "beta_humidity": round(float(beta[2]), 4),
            "beta_gas": round(float(beta[3]), 4),
            "beta_motion": round(float(beta[4]), 4)
        }

        # R²-Wert berechnen
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        self.trained_coefficients["r_squared"] = round(float(r_squared), 4)
        self.trained_coefficients["n_samples"] = len(self.training_data)
        self.trained_coefficients["trained_at"] = datetime.now().isoformat()

        self._save_calibration()

        print(f"✓ Modell trainiert (R² = {r_squared:.4f}, n = {len(self.training_data)})")
        print(f"  β₀ = {beta[0]:.2f} (Intercept)")
        print(f"  β₁ = {beta[1]:.2f} (ΔTemperatur)")
        print(f"  β₂ = {beta[2]:.2f} (ΔFeuchtigkeit)")
        print(f"  β₃ = {beta[3]:.2f} (Gas-Ratio)")
        print(f"  β₄ = {beta[4]:.2f} (Bewegung)")

        return self.trained_coefficients

    # ──────────────────────────────────────────────────────────────────────
    # Klimaanlagen-Empfehlung
    # ──────────────────────────────────────────────────────────────────────

    def _climate_recommendation(self, persons):
        """
        Empfiehlt die Klimaanlagenstufe basierend auf der Personenzahl.
        
        Stufe 1: 0–20 Personen   (minimal)
        Stufe 2: 21–45 Personen  (niedrig)
        Stufe 3: 46–70 Personen  (mittel)  ← aktueller Dauerbetrieb
        Stufe 4: 71–95 Personen  (hoch)
        Stufe 5: 96–120 Personen (maximal)
        """
        if persons <= 20:
            level = 1
            label = "Minimal"
        elif persons <= 45:
            level = 2
            label = "Niedrig"
        elif persons <= 70:
            level = 3
            label = "Mittel"
        elif persons <= 95:
            level = 4
            label = "Hoch"
        else:
            level = 5
            label = "Maximal"

        return {
            "level": level,
            "label": label,
            "persons_range": f"{max(0, (level-1)*25 - 4)}–{min(120, level*25 - 5) if level < 5 else 120}",
            "note": (
                "⚡ Klimaanlage läuft dauerhaft auf Stufe 3. "
                f"Empfohlene Stufe basierend auf ~{persons} Personen: Stufe {level} ({label})."
                + (" → Stufe reduzieren spart Energie!" if level < 3 else "")
                + (" → Stufe erhöhen empfohlen!" if level > 3 else "")
            )
        }

    # ──────────────────────────────────────────────────────────────────────
    # Hilfsmethoden
    # ──────────────────────────────────────────────────────────────────────

    def get_status(self):
        """Gibt den aktuellen Status des Estimators zurück."""
        return {
            "baseline": self.baseline,
            "model_type": "trained_regression" if self.trained_coefficients else "physical",
            "training_samples": len(self.training_data),
            "coefficients": self.trained_coefficients,
            "min_samples_for_training": 10,
            "ready_for_training": len(self.training_data) >= 10
        }

    def get_movement_rate(self, cursor, minutes=30):
        """
        Berechnet die Bewegungsrate der letzten X Minuten aus der Datenbank.
        
        Args:
            cursor: MariaDB-Cursor
            minutes: Zeitfenster in Minuten
        
        Returns:
            float: Anteil der Messungen mit Bewegung (0.0–1.0)
        """
        try:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN movement_detected = 1 THEN 1 ELSE 0 END) as motion_count
                FROM sensor_data
                WHERE timestamp >= NOW() - INTERVAL %s MINUTE
            """, (minutes,))
            
            row = cursor.fetchone()
            if row and row[0] > 0:  # row[0] = total
                total = row[0] if isinstance(row, tuple) else row["total"]
                motion = row[1] if isinstance(row, tuple) else row["motion_count"]
                return motion / total if total > 0 else 0.0
            return 0.0
        except Exception as e:
            print(f"⚠ Fehler bei Bewegungsrate: {e}")
            return 0.0


# ==============================================================================
# STANDALONE TEST
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   PERSONENSCHÄTZUNG – Regressionsanalyse (Test)")
    print("=" * 60)

    estimator = PersonEstimator()

    # Baseline setzen (leeres Restaurant)
    estimator.set_baseline(
        temperature=22.0,
        humidity=40.0,
        gas_resistance=200000
    )

    print("\n--- Testszenarien ---\n")

    # Szenario 1: Leeres Restaurant
    result = estimator.estimate(
        temperature=22.0, humidity=40.0, 
        gas_resistance=200000, movement_detected=False
    )
    print(f"Leeres Restaurant:     ~{result['estimated_persons']} Personen "
          f"(Konfidenz: {result['confidence']}%)")
    print(f"  Klima: {result['climate_recommendation']['note']}\n")

    # Szenario 2: Wenige Gäste (~20)
    result = estimator.estimate(
        temperature=23.0, humidity=43.0,
        gas_resistance=170000, movement_detected=True, movement_rate=0.3
    )
    print(f"Wenige Gäste (~20):    ~{result['estimated_persons']} Personen "
          f"(Konfidenz: {result['confidence']}%)")
    print(f"  Klima: {result['climate_recommendation']['note']}\n")

    # Szenario 3: Halbes Restaurant (~60)
    result = estimator.estimate(
        temperature=25.0, humidity=49.0,
        gas_resistance=110000, movement_detected=True, movement_rate=0.6
    )
    print(f"Halbes Restaurant:     ~{result['estimated_persons']} Personen "
          f"(Konfidenz: {result['confidence']}%)")
    print(f"  Klima: {result['climate_recommendation']['note']}\n")

    # Szenario 4: Volles Restaurant (~100)
    result = estimator.estimate(
        temperature=27.5, humidity=55.0,
        gas_resistance=60000, movement_detected=True, movement_rate=0.85
    )
    print(f"Volles Restaurant:     ~{result['estimated_persons']} Personen "
          f"(Konfidenz: {result['confidence']}%)")
    print(f"  Klima: {result['climate_recommendation']['note']}\n")

    # Szenario 5: Überfüllt (~120)
    result = estimator.estimate(
        temperature=29.0, humidity=62.0,
        gas_resistance=35000, movement_detected=True, movement_rate=0.95
    )
    print(f"Überfüllt:             ~{result['estimated_persons']} Personen "
          f"(Konfidenz: {result['confidence']}%)")
    print(f"  Klima: {result['climate_recommendation']['note']}\n")

    print("=" * 60)
    print("  Status:", json.dumps(estimator.get_status(), indent=2, default=str))