// Asia Restaurant - CPS Sensor Dashboard (Chart.js)
// Personenschaetzung (VOC) + Lineare Regression (Personen -> Temperatur)

const REFRESH_INTERVAL = 30000;

let state = {
    page: 1,
    perPage: 20,
    charts: {}
};

// ── Chart.js Defaults ──
Chart.defaults.color = '#8b8fa3';
Chart.defaults.borderColor = '#2a2d3a';
Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";

// ── Tab Navigation ──
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
});

// ── Pagination ──
document.getElementById('btnPrev').addEventListener('click', () => {
    if (state.page > 1) { state.page--; loadTable(); }
});
document.getElementById('btnNext').addEventListener('click', () => {
    state.page++;
    loadTable();
});

// ── Retrain Button ──
document.getElementById('btnRetrain').addEventListener('click', async () => {
    const status = document.getElementById('retrainStatus');
    status.textContent = 'Training laeuft...';
    try {
        const res = await fetch('/api/regression/train', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            status.textContent = 'Training erfolgreich!';
            loadRegression();
        } else {
            status.textContent = 'Fehler: ' + (data.error || 'Unbekannt');
        }
    } catch (e) {
        status.textContent = 'Verbindungsfehler';
    }
});

// ── API Helper ──
async function fetchApi(endpoint) {
    try {
        const res = await fetch(endpoint);
        return await res.json();
    } catch (e) {
        console.error('API Fehler:', endpoint, e);
        return null;
    }
}

// ── Status Update ──
function setStatus(online) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    dot.className = 'status-dot ' + (online ? 'online' : 'offline');
    text.textContent = online ? 'Verbunden' : 'Keine Verbindung';
}

// ── Create/Update Line Chart ──
function createLineChart(canvasId, labels, datasets, yTitle) {
    if (state.charts[canvasId]) {
        state.charts[canvasId].data.labels = labels;
        datasets.forEach((ds, i) => {
            state.charts[canvasId].data.datasets[i].data = ds.data;
        });
        state.charts[canvasId].update('none');
        return;
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    state.charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: datasets.length > 1, position: 'top' },
                tooltip: { backgroundColor: '#1a1d27', borderColor: '#2a2d3a', borderWidth: 1 }
            },
            scales: {
                x: { ticks: { maxTicksLimit: 12, maxRotation: 0 }, grid: { display: false } },
                y: { title: { display: !!yTitle, text: yTitle || '' }, beginAtZero: false }
            },
            elements: {
                point: { radius: 0, hoverRadius: 4 },
                line: { tension: 0.3 }
            }
        }
    });
}

// ── Format Timestamp ──
function formatTime(ts) {
    if (!ts) return '--';
    const d = new Date(ts.replace(' ', 'T'));
    return d.toLocaleTimeString('de-DE', { hour: '2-digit', minute: '2-digit' });
}

// ── Load Dashboard ──
async function loadDashboard() {
    const occ = await fetchApi('/api/occupancy/current');

    if (occ && occ.success) {
        setStatus(true);
        const d = occ.data;
        document.getElementById('currentOccupancy').textContent = d.estimated_persons ?? '--';
        document.getElementById('currentOccPercent').textContent =
            (d.occupancy_percent ?? 0).toFixed(1) + ' % Auslastung';
        if (d.sensors) {
            document.getElementById('currentTemp').textContent =
                d.sensors.temperature != null ? d.sensors.temperature.toFixed(1) + ' °C' : '--';
            document.getElementById('currentHumidity').textContent =
                d.sensors.humidity != null ? d.sensors.humidity.toFixed(1) + ' %' : '--';
            document.getElementById('currentVOC').textContent =
                d.sensors.gas_resistance != null ? Math.round(d.sensors.gas_resistance).toLocaleString() + ' Ω' : '--';
        }
    } else {
        setStatus(false);
    }

    document.getElementById('lastUpdate').textContent =
        new Date().toLocaleTimeString('de-DE');
}

// ── Load Dashboard Charts ──
async function loadDashboardCharts() {
    const data = await fetchApi('/api/data/history?hours=24&limit=500');
    if (!data || !data.success || !data.data.length) return;

    const rows = data.data;
    const labels = rows.map(r => formatTime(r.timestamp));

    // Temperaturverlauf (Hauptdiagramm laut PDF)
    createLineChart('chartTemperature', labels, [{
        label: 'Temperatur (°C)',
        data: rows.map(r => r.temperature),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        fill: true
    }], '°C');

    // Luftfeuchtigkeit
    createLineChart('chartHumidity', labels, [{
        label: 'Luftfeuchtigkeit (%)',
        data: rows.map(r => r.humidity),
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true
    }], '%');

    // Personenanzahl
    createLineChart('chartOccupancy', labels, [{
        label: 'Geschaetzte Personen',
        data: rows.map(r => r.estimated_occupancy ?? 0),
        borderColor: '#d4a853',
        backgroundColor: 'rgba(212, 168, 83, 0.1)',
        fill: true
    }], 'Personen');
}

// ── Load Regression Tab ──
async function loadRegression() {
    const [regStatus, scatter] = await Promise.all([
        fetchApi('/api/regression/status'),
        fetchApi('/api/regression/scatter?hours=48')
    ]);

    // Regression Status
    if (regStatus && regStatus.success) {
        const s = regStatus.data;
        if (s.trained) {
            document.getElementById('regSlope').textContent = s.slope.toFixed(4);
            document.getElementById('regIntercept').textContent = s.intercept.toFixed(2) + ' °C';
            document.getElementById('regR2').textContent = s.r_squared.toFixed(4);
            document.getElementById('regR2Detail').textContent =
                (s.r_squared * 100).toFixed(1) + '% der Variation erklaert';
            document.getElementById('regSamples').textContent = s.n_samples;
            document.getElementById('regTrainedAt').textContent =
                s.trained_at ? 'Trainiert: ' + new Date(s.trained_at).toLocaleString('de-DE') : '';
            document.getElementById('regFormula').textContent =
                `y = ${s.slope.toFixed(4)} · x + ${s.intercept.toFixed(2)}`;
        } else {
            document.getElementById('regSlope').textContent = '--';
            document.getElementById('regIntercept').textContent = '--';
            document.getElementById('regR2').textContent = '--';
            document.getElementById('regR2Detail').textContent = 'Noch nicht trainiert';
            document.getElementById('regSamples').textContent = '0';
        }
    }

    // Scatter Plot
    if (scatter && scatter.success) {
        const points = scatter.data.points;
        const regLine = scatter.data.regression_line;

        const datasets = [{
            label: 'Messpunkte',
            data: points,
            type: 'scatter',
            backgroundColor: 'rgba(212, 168, 83, 0.6)',
            borderColor: '#d4a853',
            pointRadius: 4,
            pointHoverRadius: 6
        }];

        // Regressionslinie hinzufuegen
        if (regLine) {
            datasets.push({
                label: `Regression (R² = ${regLine.r_squared.toFixed(4)})`,
                data: regLine.points,
                type: 'scatter',
                showLine: true,
                borderColor: '#c41e3a',
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            });
        }

        if (state.charts['chartScatter']) {
            state.charts['chartScatter'].data.datasets = datasets;
            state.charts['chartScatter'].update();
        } else {
            const ctx = document.getElementById('chartScatter').getContext('2d');
            state.charts['chartScatter'] = new Chart(ctx, {
                type: 'scatter',
                data: { datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top' },
                        tooltip: {
                            backgroundColor: '#1a1d27',
                            borderColor: '#2a2d3a',
                            borderWidth: 1,
                            callbacks: {
                                label: (ctx) =>
                                    `${ctx.dataset.label}: ${ctx.parsed.x} Personen → ${ctx.parsed.y.toFixed(1)} °C`
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: { display: true, text: 'Geschaetzte Personenanzahl' },
                            min: 0, max: 130
                        },
                        y: {
                            title: { display: true, text: 'Temperatur (°C)' }
                        }
                    }
                }
            });
        }

        // Vorhersagen
        const scenarios = scatter.data.scenarios;
        const grid = document.getElementById('predictionsGrid');
        if (scenarios) {
            grid.innerHTML = scenarios.map(s => `
                <div class="prediction-card">
                    <div class="persons">${s.persons}</div>
                    <div class="label">${s.label}</div>
                    <div class="temp">${s.predicted_temp.toFixed(1)} <span class="temp-unit">°C</span></div>
                </div>
            `).join('');
        } else {
            grid.innerHTML = '<p>Modell noch nicht trainiert. Druecke "Modell neu trainieren".</p>';
        }
    }
}

// ── Load Sensors Tab ──
async function loadSensors() {
    const [occ, stats] = await Promise.all([
        fetchApi('/api/occupancy/current'),
        fetchApi('/api/data/stats')
    ]);

    if (occ && occ.success && occ.data.sensors) {
        const s = occ.data.sensors;
        document.getElementById('sensorTemp').textContent =
            s.temperature != null ? s.temperature.toFixed(1) + ' °C' : '--';
        document.getElementById('sensorHumidity').textContent =
            s.humidity != null ? s.humidity.toFixed(1) + ' %' : '--';
        document.getElementById('sensorGas').textContent =
            s.gas_resistance != null ? Math.round(s.gas_resistance).toLocaleString() + ' Ω' : '--';
        document.getElementById('sensorMovement').textContent =
            s.movement_detected ? 'Ja' : 'Nein';
        document.getElementById('sensorMovementStatus').textContent =
            s.movement_detected ? 'Bewegung erkannt' : 'Keine Bewegung';
    }

    if (stats && stats.success) {
        const d = stats.data;
        document.getElementById('sensorTempRange').textContent =
            d.min_temp != null ? `Min ${d.min_temp.toFixed(1)} / Max ${d.max_temp.toFixed(1)} °C` : '--';
        document.getElementById('sensorHumidityAvg').textContent =
            d.avg_humidity != null ? 'Durchschnitt: ' + d.avg_humidity.toFixed(1) + ' %' : '--';
    }

    // Gas and Pressure charts
    const hist = await fetchApi('/api/data/history?hours=24&limit=500');
    if (hist && hist.success && hist.data.length) {
        const rows = hist.data;
        const labels = rows.map(r => formatTime(r.timestamp));

        createLineChart('chartGas', labels, [{
            label: 'VOC / Gasresistenz (Ohm)',
            data: rows.map(r => r.gas_resistance),
            borderColor: '#d4a853',
            backgroundColor: 'rgba(212, 168, 83, 0.1)',
            fill: true
        }], 'Ohm');

        createLineChart('chartPressure', labels, [{
            label: 'Luftdruck (hPa)',
            data: rows.map(r => r.pressure),
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            fill: true
        }], 'hPa');
    }
}

// ── Load History Table ──
async function loadTable() {
    const data = await fetchApi(`/api/data/table?page=${state.page}&per_page=${state.perPage}`);
    if (!data || !data.success) return;

    const tbody = document.getElementById('tableBody');
    if (!data.data.length) {
        tbody.innerHTML = '<tr><td colspan="7">Keine Daten vorhanden</td></tr>';
        return;
    }

    tbody.innerHTML = data.data.map(r => `
        <tr>
            <td>${r.id}</td>
            <td>${r.timestamp || '--'}</td>
            <td>${r.temperature != null ? r.temperature.toFixed(1) : '--'}</td>
            <td>${r.humidity != null ? r.humidity.toFixed(1) : '--'}</td>
            <td>${r.gas_resistance != null ? Math.round(r.gas_resistance).toLocaleString() : '--'}</td>
            <td><span class="badge ${r.movement_detected ? 'badge-yes' : 'badge-no'}">${r.movement_detected ? 'Ja' : 'Nein'}</span></td>
            <td>${r.estimated_occupancy ?? '--'}</td>
        </tr>
    `).join('');

    const p = data.pagination;
    document.getElementById('pageInfo').textContent = `Seite ${p.page} von ${p.pages}`;
    document.getElementById('tableInfo').textContent = `${p.total} Eintraege`;
    document.getElementById('btnPrev').disabled = p.page <= 1;
    document.getElementById('btnNext').disabled = p.page >= p.pages;
}

// ── Initial Load ──
async function init() {
    await Promise.all([
        loadDashboard(),
        loadDashboardCharts(),
        loadTable()
    ]);
    loadRegression();
    loadSensors();
}

init();

// ── Auto-refresh every 30s ──
setInterval(() => {
    loadDashboard();
    loadDashboardCharts();
    loadRegression();
    loadSensors();
}, REFRESH_INTERVAL);
