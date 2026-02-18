# Raspberry Pi Sensorstation – BME680 + PIR mit MariaDB-Anbindung

import sys, time
from datetime import datetime
import pymysql

try:
    import RPi.GPIO as GPIO
    from rpi_sensors.sensors import alle_sensoren_auslesen
    from rpi_sensors.motion_sensor import bewegung_ueberwachen
    from rpi_sensors.bme680_sensor import messen_intervall
    SENSORS_AVAILABLE = True
except (ImportError, RuntimeError):
    SENSORS_AVAILABLE = False
    print("[WARNUNG] Raspberry Pi Sensoren nicht verfuegbar (kein RPi-System erkannt)")

DB_CONFIG = {'host':'localhost','port':3306,'user':'root','password':'root','database':'sensor_db','cursorclass':pymysql.cursors.Cursor}
MESSINTERVALL = 300


def db_verbinden():
    try: return pymysql.connect(**DB_CONFIG)
    except pymysql.MySQLError as e: print(f"DB-Fehler: {e}"); sys.exit(1)


def tabelle_erstellen(cursor):
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


def daten_speichern(cursor, conn, daten):
    if daten['temperatur'] is None:
        print("Keine Sensordaten – übersprungen")
        return
    cursor.execute(
        "INSERT INTO sensor_data (timestamp, temperature, pressure, humidity, gas_resistance, movement_detected) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (datetime.now(), daten['temperatur'], daten['druck'], daten['feuchtigkeit'], daten['gas'], daten['bewegung'])
    )
    conn.commit()
    print("Daten gespeichert")


def ausgabe(d):
    gas = f"{d['gas']} Ω" if d['gas'] else "(aufheizen)"
    print(f"{d['temperatur']} °C | {d['druck']} hPa | {d['feuchtigkeit']} %RH | Gas: {gas}")


def hauptschleife(cursor, conn):
    try:
        while True:
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Lese Sensoren...")
            daten = alle_sensoren_auslesen()
            daten_speichern(cursor, conn, daten)
            print(f"Nächste Messung in {MESSINTERVALL}s")
            time.sleep(MESSINTERVALL)
    except KeyboardInterrupt:
        print("\nMessung beendet.")


if __name__ == "__main__":
    if not SENSORS_AVAILABLE:
        print("Dieses Skript benoetigt einen Raspberry Pi mit angeschlossenen Sensoren.")
        print("Zum Starten des Dashboards bitte 'app.py' ausfuehren.")
        sys.exit(1)

    conn = db_verbinden()
    cursor = conn.cursor()
    tabelle_erstellen(cursor)
    conn.commit()

    print("\n  1 - Bewegungssensor (PIR)\n  2 - Umweltsensor (BME680)\n  3 - Alle Sensoren + Datenbank\n  0 - Beenden\n")
    wahl = input("Auswahl: ").strip()

    aktionen = {
        "1": bewegung_ueberwachen,
        "2": lambda: messen_intervall(intervall=5, callback=ausgabe),
        "3": lambda: hauptschleife(cursor, conn),
        "0": lambda: print("Beendet.")
    }

    aktionen.get(wahl, lambda: print("Ungültige Eingabe"))()

    GPIO.cleanup()
    try: conn.close()
    except: pass