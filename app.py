from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pymysql
import sys
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Erlaubt API-Zugriffe

# DATENBANK-KONFIGURATION
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'sensor_db',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}


def get_db_connection():
    """Erstellt eine neue Datenbankverbindung."""
    try:
        conn = pymysql.connect(**db_config)
        return conn
    except pymysql.Error as e:
        print(f"Fehler bei Datenbankverbindung: {e}")
        return None


# HAUPTSEITE

@app.route("/")
def index():
    """Hauptseite – Dashboard."""
    return render_template("index.html")

# API ENDPUNKTE
@app.route("/api/data/latest")
def api_latest():
    """Neueste Messung."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "error": "Datenbankverbindung fehlgeschlagen"}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM sensor_data 
            ORDER BY id DESC 
            LIMIT 1
        """)
        data = cursor.fetchone()
        
        if data and data.get("timestamp"):
            data["timestamp"] = data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({"success": True, "data": data or {}})
    except pymysql.Error as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/data/stats")
def api_stats():
    """Statistiken der letzten 24 Stunden."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "error": "Datenbankverbindung fehlgeschlagen"}), 500

    try:
        cursor = conn.cursor()
        
        # 24 Stunden zurück
        time_24h_ago = datetime.now() - timedelta(hours=24)
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_readings,
                AVG(temperature) as avg_temp,
                MAX(temperature) as max_temp,
                MIN(temperature) as min_temp,
                AVG(humidity) as avg_humidity,
                AVG(pressure) as avg_pressure,
                SUM(CASE WHEN movement_detected = 1 THEN 1 ELSE 0 END) as movement_count
            FROM sensor_data
            WHERE timestamp >= %s
        """, (time_24h_ago,))
        
        stats = cursor.fetchone()
        
        return jsonify({"success": True, "data": stats or {}})
    except pymysql.Error as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/data/history")
def api_history():
    """Historische Daten für Charts."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "error": "Datenbankverbindung fehlgeschlagen"}), 500

    try:
        hours = int(request.args.get('hours', 24))
        limit = int(request.args.get('limit', 1000))
        
        time_ago = datetime.now() - timedelta(hours=hours)
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM sensor_data
            WHERE timestamp >= %s
            ORDER BY timestamp ASC
            LIMIT %s
        """, (time_ago, limit))
        
        data = cursor.fetchall()
        
        # Timestamps formatieren
        for row in data:
            if row.get("timestamp"):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({"success": True, "data": data})
    except pymysql.Error as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/data/table")
def api_table():
    """Paginierte Tabellendaten."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"success": False, "error": "Datenbankverbindung fehlgeschlagen"}), 500

    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        offset = (page - 1) * per_page
        
        cursor = conn.cursor()
        
        # Gesamtanzahl
        cursor.execute("SELECT COUNT(*) as total FROM sensor_data")
        total = cursor.fetchone()['total']
        
        # Daten für aktuelle Seite
        cursor.execute("""
            SELECT * FROM sensor_data
            ORDER BY id DESC
            LIMIT %s OFFSET %s
        """, (per_page, offset))
        
        data = cursor.fetchall()
        
        # Timestamps formatieren
        for row in data:
            if row.get("timestamp"):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        
        pagination = {
            'page': page,
            'per_page': per_page,
            'total': total,
            'pages': (total + per_page - 1) // per_page
        }
        
        return jsonify({
            "success": True,
            "data": data,
            "pagination": pagination
        })
    except pymysql.Error as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/api/data")
def api_data_legacy():
    """Legacy-Endpunkt (falls noch genutzt)."""
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Datenbankverbindung fehlgeschlagen"}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensor_data ORDER BY id DESC LIMIT 50")
        data = cursor.fetchall()

        # datetime-Objekte in Strings umwandeln für JSON
        for row in data:
            if row.get("timestamp"):
                row["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    except pymysql.Error as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

    return jsonify(data)


# SERVER STARTEN

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   RASPBERRY PI SENSOR DASHBOARD - Flask Server")
    print("   " + "=" * 56)
    print("   Dashboard:  http://0.0.0.0:5000")
    print("   Lokal:      http://localhost:5000")
    print("   📡 API:        http://0.0.0.0:5000/api/data/latest")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)