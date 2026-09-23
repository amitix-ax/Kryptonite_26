/* Iceberg Hazard Overlay — Small Iceberg Detection & Trajectory Visualization */

class IcebergHazardOverlay {
  constructor(mapCtrl) {
    this.mapCtrl = mapCtrl;
    this.icebergMarkers = [];
    this.trajectoryLines = [];
    this.uncertaintyCones = [];
    this.detectedIcebergs = [];
    this.envOverlayPoints = [];
    this.apiBase = '';
    this.isLoading = false;
  }

  // ── Detect & Display All Icebergs ──────────────────────
  async loadIcebergs(timestamp = null, seed = null) {
    if (this.isLoading) return;
    this.isLoading = true;
    this._updateStatusBadge('scanning', 'Scanning...');

    try {
      const body = {
        n_population: 30,
        detection_sensitivity: 0.6,
      };
      if (timestamp) body.timestamp = timestamp;
      if (seed) body.seed = seed;

      const res = await fetch(`${this.apiBase}/api/icebergs/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();

      this.detectedIcebergs = data.icebergs || [];
      this._clearMarkers();
      this._renderIcebergs(this.detectedIcebergs);
      this._updateStats(data);
      this._updateStatusBadge('active', `${data.total_icebergs} tracked`);

    } catch (err) {
      console.error('[IcebergOverlay] Detection failed:', err);
      this._updateStatusBadge('error', 'Detection failed');
    }
    this.isLoading = false;
  }

  // ── Predict Trajectory for a Single Iceberg ────────────
  async predictTrajectory(iceberg) {
    try {
      const res = await fetch(`${this.apiBase}/api/icebergs/predict-trajectory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          iceberg_id: iceberg.id,
          lat: iceberg.lat,
          lon: iceberg.lon,
          length_m: iceberg.length_m || iceberg.estimated_length_m || 50,
          height_m: iceberg.height_m || 10,
          forecast_hours: 48,
          dt_hours: 3,
        }),
      });
      const data = await res.json();

      if (data.trajectory) {
        this._renderTrajectory(iceberg, data.trajectory);
      }
      return data;
    } catch (err) {
      console.error('[IcebergOverlay] Trajectory prediction failed:', err);
      return null;
    }
  }

  // ── Load Environment Visualization ─────────────────────
  async loadEnvironment(timestamp = null) {
    try {
      const body = {};
      if (timestamp) body.timestamp = timestamp;

      const res = await fetch(`${this.apiBase}/api/environment/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();

      if (data.sample_points) {
        this._renderEnvironmentOverlay(data.sample_points);
      }
      this._updateEnvironmentPanel(data);
      return data;
    } catch (err) {
      console.error('[IcebergOverlay] Environment load failed:', err);
      return null;
    }
  }

  // ── Run Scenario Test ──────────────────────────────────
  async runScenario(scenarioKey) {
    const statusEl = document.getElementById('scenarioStatus');
    if (statusEl) statusEl.textContent = 'Running...';

    try {
      const res = await fetch(`${this.apiBase}/api/scenarios/run/${scenarioKey}`, {
        method: 'POST',
      });
      const report = await res.json();
      this._renderScenarioReport(report);
      return report;
    } catch (err) {
      console.error('[IcebergOverlay] Scenario failed:', err);
      if (statusEl) statusEl.textContent = 'Failed';
      return null;
    }
  }

  // ── Internal: Render Icebergs on Map ───────────────────
  _renderIcebergs(icebergs) {
    const map = this.mapCtrl.map;
    if (!map) return;

    icebergs.forEach(berg => {
      if (!berg.lat || !berg.lon) return;

      const hazardLevel = berg.hazard_level || 'LOW';
      const colors = {
        CRITICAL: { fill: '#f43f5e', border: '#be123c', glow: 'rgba(244,63,94,0.4)' },
        HIGH: { fill: '#f59e0b', border: '#d97706', glow: 'rgba(245,158,11,0.3)' },
        MODERATE: { fill: '#38bdf8', border: '#0284c7', glow: 'rgba(56,189,248,0.3)' },
        LOW: { fill: '#94a3b8', border: '#64748b', glow: 'rgba(148,163,184,0.2)' },
      };
      const c = colors[hazardLevel] || colors.LOW;

      const sizeMap = { growler: 6, bergy_bit: 8, small: 10, medium: 14, large: 18 };
      const size = sizeMap[berg.size_class] || 10;

      const icon = L.divIcon({
        className: '',
        html: `<div style="
          width:${size}px; height:${size}px;
          background:${c.fill};
          border:2px solid ${c.border};
          border-radius:${berg.size_class === 'growler' ? '50%' : '3px'};
          box-shadow:0 0 ${size}px ${c.glow};
          cursor:pointer;
          transition: transform 0.2s ease;
        " onmouseover="this.style.transform='scale(1.5)'" onmouseout="this.style.transform='scale(1)'"></div>`,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
      });

      const marker = L.marker([berg.lat, berg.lon], { icon }).addTo(map);

      // Popup with iceberg details
      const popupContent = this._buildPopup(berg);
      marker.bindPopup(popupContent, {
        maxWidth: 300,
        className: 'iceberg-popup',
      });

      // Click to predict trajectory
      marker.on('click', () => {
        this.predictTrajectory(berg);
      });

      this.icebergMarkers.push(marker);
    });
  }

  _buildPopup(berg) {
    const hazardBadge = {
      CRITICAL: '<span style="color:#f43f5e;font-weight:700;">⬤ CRITICAL</span>',
      HIGH: '<span style="color:#f59e0b;font-weight:700;">⬤ HIGH</span>',
      MODERATE: '<span style="color:#38bdf8;font-weight:700;">⬤ MODERATE</span>',
      LOW: '<span style="color:#94a3b8;font-weight:700;">⬤ LOW</span>',
    };

    return `
      <div style="font-family:'Inter',sans-serif;font-size:11px;line-height:1.6;">
        <div style="font-size:13px;font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:8px;">
          🧊 ${berg.id}
          ${hazardBadge[berg.hazard_level] || ''}
        </div>
        <table style="width:100%;font-size:10px;border-collapse:collapse;">
          <tr><td style="color:#94a3b8;padding:2px 6px 2px 0;">Class</td><td style="font-weight:600;">${berg.size_class}</td></tr>
          <tr><td style="color:#94a3b8;padding:2px 6px 2px 0;">Length</td><td>${berg.length_m || berg.estimated_length_m || '?'}m</td></tr>
          ${berg.height_m ? `<tr><td style="color:#94a3b8;padding:2px 6px 2px 0;">Height</td><td>${berg.height_m}m</td></tr>` : ''}
          ${berg.draft_m ? `<tr><td style="color:#94a3b8;padding:2px 6px 2px 0;">Draft</td><td>${berg.draft_m}m</td></tr>` : ''}
          ${berg.mass_tonnes ? `<tr><td style="color:#94a3b8;padding:2px 6px 2px 0;">Mass</td><td>${Math.round(berg.mass_tonnes).toLocaleString()} t</td></tr>` : ''}
          <tr><td style="color:#94a3b8;padding:2px 6px 2px 0;">Position</td><td style="font-family:monospace;">${berg.lat.toFixed(3)}°, ${berg.lon.toFixed(3)}°</td></tr>
          ${berg.speed_ms ? `<tr><td style="color:#94a3b8;padding:2px 6px 2px 0;">Speed</td><td>${berg.speed_ms.toFixed(3)} m/s</td></tr>` : ''}
          ${berg.confidence ? `<tr><td style="color:#94a3b8;padding:2px 6px 2px 0;">Confidence</td><td>${(berg.confidence * 100).toFixed(0)}%</td></tr>` : ''}
          ${berg.source_hotspot ? `<tr><td style="color:#94a3b8;padding:2px 6px 2px 0;">Source</td><td>${berg.source_hotspot}</td></tr>` : ''}
        </table>
        <div style="margin-top:6px;font-size:9px;color:#64748b;">Click to predict trajectory</div>
      </div>
    `;
  }

  // ── Internal: Render Trajectory on Map ─────────────────
  _renderTrajectory(iceberg, trajectory) {
    const map = this.mapCtrl.map;
    if (!map || !trajectory || trajectory.length < 2) return;

    // Clear previous trajectories
    this._clearTrajectories();

    const points = trajectory.map(p => [p.lat, p.lon]);
    const hazardLevel = iceberg.hazard_level || 'MODERATE';
    const color = {
      CRITICAL: '#f43f5e',
      HIGH: '#f59e0b',
      MODERATE: '#38bdf8',
      LOW: '#94a3b8',
    }[hazardLevel] || '#38bdf8';

    // Predicted path line
    const line = L.polyline(points, {
      color: color,
      weight: 3,
      dashArray: '8,6',
      opacity: 0.8,
    }).addTo(map);
    this.trajectoryLines.push(line);

    // Uncertainty cone at the last point
    const last = trajectory[trajectory.length - 1];
    if (last.uncertainty_km > 0) {
      const circle = L.circle([last.lat, last.lon], {
        radius: last.uncertainty_km * 1000,
        color: color,
        fillColor: color,
        fillOpacity: 0.08,
        weight: 1,
        dashArray: '4,4',
      }).addTo(map);
      this.uncertaintyCones.push(circle);
    }

    // Time markers along trajectory
    trajectory.forEach((pt, i) => {
      if (i % 4 === 0 && i > 0) {
        const timeIcon = L.divIcon({
          className: '',
          html: `<div style="
            background:rgba(15,23,42,0.85);
            color:${color};
            font-size:8px; font-weight:600;
            padding:1px 4px;
            border-radius:3px;
            border:1px solid ${color}40;
            white-space:nowrap;
          ">+${pt.elapsed_h}h</div>`,
          iconSize: [30, 14],
          iconAnchor: [15, 7],
        });
        const tm = L.marker([pt.lat, pt.lon], { icon: timeIcon }).addTo(map);
        this.trajectoryLines.push(tm);
      }
    });

    // Fit map to show trajectory
    map.fitBounds(L.latLngBounds(points), { padding: [40, 40], maxZoom: 8 });
  }

  // ── Internal: Environment Overlay ──────────────────────
  _renderEnvironmentOverlay(samplePoints) {
    const map = this.mapCtrl.map;
    if (!map) return;

    // Clear previous
    this.envOverlayPoints.forEach(m => map.removeLayer(m));
    this.envOverlayPoints = [];

    samplePoints.forEach(pt => {
      const ice = pt.ice_conc || 0;
      const windSpd = Math.sqrt((pt.wind_u || 0) ** 2 + (pt.wind_v || 0) ** 2);

      if (ice < 0.05 && windSpd < 5) return; // Skip boring points

      // Wind vector arrow
      if (windSpd > 2) {
        const windAngle = Math.atan2(pt.wind_v, pt.wind_u) * (180 / Math.PI);
        const arrowIcon = L.divIcon({
          className: '',
          html: `<div style="
            transform:rotate(${-windAngle + 90}deg);
            color:rgba(255,255,255,0.5);
            font-size:${Math.min(20, 8 + windSpd)}px;
          ">↑</div>`,
          iconSize: [20, 20],
          iconAnchor: [10, 10],
        });
        const arrow = L.marker([pt.lat, pt.lon], { icon: arrowIcon, interactive: false }).addTo(map);
        this.envOverlayPoints.push(arrow);
      }
    });
  }

  // ── Internal: Update Stats Panel ───────────────────────
  _updateStats(data) {
    const el = document.getElementById('icebergStatsPanel');
    if (!el) return;

    const dist = data.size_distribution || {};
    const hazards = data.hazard_distribution || {};

    el.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <div class="tele-card">
          <div class="tele-label">Total Detected</div>
          <div class="tele-val">${data.total_icebergs || 0}</div>
        </div>
        <div class="tele-card">
          <div class="tele-label">Env. Detected</div>
          <div class="tele-val">${data.detected_from_environment || 0}</div>
        </div>
      </div>
      <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">
        ${Object.entries(hazards).map(([level, count]) => {
          const colors = { CRITICAL: '#f43f5e', HIGH: '#f59e0b', MODERATE: '#38bdf8', LOW: '#94a3b8' };
          return `<span style="
            font-size:10px;font-weight:600;
            padding:2px 8px;border-radius:4px;
            background:${colors[level]}20;
            color:${colors[level]};
            border:1px solid ${colors[level]}40;
          ">${level}: ${count}</span>`;
        }).join('')}
      </div>
      <div style="margin-top:6px;font-size:9px;color:var(--text3);">
        Size: ${Object.entries(dist).map(([c, n]) => `${c}:${n}`).join(' · ')}
      </div>
    `;
  }

  _updateEnvironmentPanel(data) {
    const el = document.getElementById('envStatsPanel');
    if (!el || !data.fields) return;

    const f = data.fields;
    el.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        <div class="tele-card">
          <div class="tele-label">Wind Speed</div>
          <div class="tele-val">${Math.sqrt(f.wind_u.mean**2 + f.wind_v.mean**2).toFixed(1)}<span class="tele-unit">m/s</span></div>
        </div>
        <div class="tele-card">
          <div class="tele-label">SST</div>
          <div class="tele-val">${f.sst.mean.toFixed(1)}<span class="tele-unit">°C</span></div>
        </div>
        <div class="tele-card">
          <div class="tele-label">Ice Conc.</div>
          <div class="tele-val">${(f.ice_conc.mean * 100).toFixed(0)}<span class="tele-unit">%</span></div>
        </div>
        <div class="tele-card">
          <div class="tele-label">Wave Ht.</div>
          <div class="tele-val">${f.wave_height.mean.toFixed(1)}<span class="tele-unit">m</span></div>
        </div>
      </div>
      <div style="margin-top:4px;font-size:9px;color:var(--text3);">
        Season factor: ${data.season_factor?.toFixed(2) || 'N/A'} · 
        Max ice: ${(f.ice_conc.max * 100).toFixed(0)}%
      </div>
    `;
  }

  _renderScenarioReport(report) {
    const el = document.getElementById('scenarioResults');
    if (!el) return;

    const phases = report.phases || {};
    const timing = report.timing || {};
    const hazards = phases.hazard_assessment || {};

    el.innerHTML = `
      <div style="font-size:12px;font-weight:700;color:#fff;margin-bottom:8px;">${report.name}</div>
      <div style="font-size:10px;color:var(--text2);margin-bottom:8px;">${report.description}</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">
        ${Object.entries(hazards.distribution || {}).map(([level, count]) => {
          const colors = { CRITICAL: '#f43f5e', HIGH: '#f59e0b', MODERATE: '#38bdf8', LOW: '#94a3b8' };
          return `<span style="font-size:9px;font-weight:600;padding:2px 6px;border-radius:3px;
            background:${colors[level]}20;color:${colors[level]};border:1px solid ${colors[level]}40;
          ">${level}: ${count}</span>`;
        }).join('')}
      </div>
      <div style="font-size:10px;color:var(--status-amber);font-weight:600;margin-bottom:4px;">
        ${hazards.navigation_advisory || 'No advisory'}
      </div>
      <div style="font-size:9px;color:var(--text3);display:flex;gap:12px;">
        <span>Detection: ${timing.detection_s?.toFixed(2) || '?'}s</span>
        <span>Trajectory: ${timing.trajectory_prediction_s?.toFixed(2) || '?'}s</span>
        <span>Total: ${timing.total_s?.toFixed(2) || '?'}s</span>
      </div>
    `;
  }

  _updateStatusBadge(status, text) {
    const el = document.getElementById('icebergStatusBadge');
    if (!el) return;
    const styles = {
      scanning: 'background:rgba(245,158,11,0.2);color:#f59e0b;border-color:rgba(245,158,11,0.4);',
      active: 'background:rgba(16,185,129,0.2);color:#10b981;border-color:rgba(16,185,129,0.4);',
      error: 'background:rgba(244,63,94,0.2);color:#f43f5e;border-color:rgba(244,63,94,0.4);',
    };
    el.style.cssText = `font-size:10px;font-weight:600;padding:3px 10px;border-radius:6px;border:1px solid;${styles[status] || styles.active}`;
    el.textContent = `🧊 ${text}`;
  }

  _clearMarkers() {
    const map = this.mapCtrl.map;
    if (!map) return;
    this.icebergMarkers.forEach(m => map.removeLayer(m));
    this.icebergMarkers = [];
  }

  _clearTrajectories() {
    const map = this.mapCtrl.map;
    if (!map) return;
    this.trajectoryLines.forEach(l => map.removeLayer(l));
    this.trajectoryLines = [];
    this.uncertaintyCones.forEach(c => map.removeLayer(c));
    this.uncertaintyCones = [];
  }

  clearAll() {
    this._clearMarkers();
    this._clearTrajectories();
    this.envOverlayPoints.forEach(m => this.mapCtrl.map?.removeLayer(m));
    this.envOverlayPoints = [];
  }
}
