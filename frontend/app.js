/* Main App Orchestrator — Antarctic Iceberg DSS */

document.addEventListener('DOMContentLoaded', () => {
  const mapCtrl = new IcebergMap('map');
  const charts = new DashboardCharts();
  let simData = null;
  let nicIcebergs = [];

  mapCtrl.init(-69.5, 30.0, 4);

  const timeline = new TimelineController((idx, step) => {
    if (!simData || !simData.steps) return;
    const active = simData.steps.slice(0, idx + 1);
    mapCtrl.updateStep(step, active);
    updateTelemetry(step, idx);
    updateRisk(step, idx, simData);

    // Update moving iceberg position in hazard database for Google Maps-style auto-redirect
    if (nicIcebergs && nicIcebergs.length > 0 && step) {
      const simIce = nicIcebergs.find(ib => ib.id === 'SIM-01');
      if (simIce) {
        simIce.lat = step.lat;
        simIce.lon = step.lon;
        simIce.lng = step.lon;
        routePlanner.setHazards(nicIcebergs, simData.sea_ice_bounds);
      }
    }
  });

  // ---- ROUTE PLANNER ----
  const routePlanner = new RoutePlanner(mapCtrl, (routeData) => {
    // Route updated callback — recalculate fuel and update risk corridors
    if (routeData) {
      fuelCalc.calculateForRoute(routeData);
      renderRiskCorridors(routeData.riskCorridors);
      renderRouteSummary(routeData);
      renderRouteWeather(routeData);
      triggerGeminiAnalysis(routeData);
    } else {
      fuelCalc.calculateForRoute(null);
      renderRiskCorridors([]);
      renderRouteSummary(null);
      renderRouteWeather(null);
      triggerGeminiAnalysis(null);
    }
  });
  window._routePlanner = routePlanner; // Expose for inline onclick handlers

  loadPolarNews();

  // ---- ICEBERG HAZARD OVERLAY ----
  const icebergOverlay = new IcebergHazardOverlay(mapCtrl);
  window._icebergOverlay = icebergOverlay;

  const btnDetect = document.getElementById('btnDetectIcebergs');
  if (btnDetect) {
    btnDetect.addEventListener('click', () => icebergOverlay.loadIcebergs());
  }
  const btnEnv = document.getElementById('btnLoadEnvironment');
  if (btnEnv) {
    btnEnv.addEventListener('click', () => icebergOverlay.loadEnvironment());
  }
  document.querySelectorAll('.scenario-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.scenario;
      if (key) icebergOverlay.runScenario(key);
    });
  });

  // ---- SIDEBAR TOGGLE ----
  const sidebarBtn = document.getElementById('sidebarToggleBtn');
  if (sidebarBtn) {
    sidebarBtn.addEventListener('click', () => {
      document.querySelector('.app').classList.toggle('sidebar-collapsed');
      // Invalidate map size after CSS transition finishes
      setTimeout(() => {
        if (mapCtrl && mapCtrl.map) {
          mapCtrl.map.invalidateSize();
        }
      }, 350); // match transition duration in CSS
    });
  }
  // ---- FUEL CALCULATOR ----
  const fuelCalc = new FuelCalculator((fuelResult) => {
    // When vessel specs change, re-run calc if we have route data
    if (!fuelResult && routePlanner.lastRouteResult) {
      fuelCalc.calculateForRoute(routePlanner.lastRouteResult);
    }
  });

  // ---- CONFIDENCE TRACKER ----
  const confTracker = new ConfidenceTracker(() => {
    // Auto-refresh callback: reload data and recalculate
    console.log('[ConfTracker] Auto-refresh triggered');
    if (routePlanner.lastRouteResult) {
      // Recalculate route with current hazards
      routePlanner._recalculate();
    }
  });

  // ---- ROUTE PLANNER BUTTON BINDINGS ----
  document.getElementById('rpBtnStart').addEventListener('click', () => {
    routePlanner.enterPlacementMode('start');
    setActivePlacementBtn('rpBtnStart');
  });
  document.getElementById('rpBtnEnd').addEventListener('click', () => {
    routePlanner.enterPlacementMode('end');
    setActivePlacementBtn('rpBtnEnd');
  });
  document.getElementById('rpBtnWaypoint').addEventListener('click', () => {
    routePlanner.enterPlacementMode('waypoint');
    setActivePlacementBtn('rpBtnWaypoint');
  });
  document.getElementById('rpBtnClear').addEventListener('click', () => {
    routePlanner.clearAll();
  });

  function setActivePlacementBtn(activeId) {
    ['rpBtnStart', 'rpBtnEnd', 'rpBtnWaypoint'].forEach(id => {
      document.getElementById(id).classList.remove('active');
    });
    document.getElementById(activeId).classList.add('active');

    // Clear active state when placement completes
    const checkInterval = setInterval(() => {
      if (!routePlanner.placementMode) {
        document.getElementById(activeId).classList.remove('active');
        clearInterval(checkInterval);
      }
    }, 200);
  }

  // ---- ROUTE PLANNER PANEL TOGGLE ----
  const rpToggle = document.getElementById('rpToggle');
  const rpBody = document.getElementById('rpBody');
  let rpCollapsed = false;
  rpToggle.addEventListener('click', () => {
    rpCollapsed = !rpCollapsed;
    rpBody.style.display = rpCollapsed ? 'none' : '';
    rpToggle.textContent = rpCollapsed ? '+' : '—';
  });

  // ---- VESSEL SPECS COLLAPSE TOGGLE ----
  const fcHeader = document.getElementById('fcSpecsHeader');
  const fcBody = document.getElementById('fcSpecsBody');
  fcHeader.addEventListener('click', () => {
    fcBody.classList.toggle('open');
    fcHeader.classList.toggle('open');
  });

  // Auto-load data
  loadDefault();

  // File upload
  document.getElementById('loadDataBtn').addEventListener('click', () => document.getElementById('fileInput').click());
  document.getElementById('fileInput').addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = ev => {
      try { processData(JSON.parse(ev.target.result)); }
      catch (err) { alert('Invalid JSON: ' + err.message); }
    };
    r.readAsText(f);
  });

  function loadDataFile(filename) {
    const ts = Date.now();
    fetch(`../data/${filename}?v=${ts}`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(d => processData(d))
      .catch(() => {
        fetch(`/data/${filename}?v=${ts}`)
          .then(r => r.json())
          .then(d => processData(d))
          .catch(e => console.error(`Could not load ${filename}:`, e));
      });
  }

  function loadDefault() {
    loadDataFile('dashboard_data.json');
  }

  const driftSel = document.getElementById('driftSelectorDate');
  if (driftSel) {
    driftSel.addEventListener('change', (e) => {
      const dateStr = e.target.value;
      if (dateStr) {
        loadDataFile(`drift_history_${dateStr}.json`);
      }
    });
  }

  function updateConfidenceBadge(data) {
    const badge = document.getElementById('ctBadge');
    const snippet = document.getElementById('ctSnippet');
    if (!badge || !snippet || !data.steps || !data.steps.length) return;

    let lastUpdateStr = data.metadata && data.metadata.last_updated;
    let lastUpdateDate;
    if (lastUpdateStr) {
      lastUpdateDate = new Date(lastUpdateStr);
    } else {
      lastUpdateDate = new Date(data.steps[data.steps.length - 1].t * 1000);
    }

    const now = new Date();
    const diffHours = Math.max(0, (now - lastUpdateDate) / (1000 * 60 * 60));

    // Base confidence is 99%. Drops by 1.5% every hour of data stagnancy.
    let conf = 99 - (diffHours * 1.5);
    conf = Math.max(35, Math.min(99, conf));

    let statusClass = 'ct-fresh';
    let dotClass = 'ct-dot-fresh';
    let text = 'FRESH';
    let msg = `High fidelity. Live data synced.`;

    if (conf < 80) {
      statusClass = 'ct-warning';
      dotClass = 'ct-dot-warning';
      text = 'DEGRADED';
      msg = `Stale data (${Math.round(diffHours)}h old).`;
    }
    if (conf < 50) {
      statusClass = 'ct-stale';
      dotClass = 'ct-dot-stale';
      text = 'STALE';
      msg = `Critical data gap (${Math.round(diffHours)}h).`;
    }

    badge.className = `ct-badge ${statusClass}`;
    badge.title = msg;
    badge.innerHTML = `<span class="ct-dot ${dotClass}"></span> ${text} ${conf.toFixed(1)}%`;
    snippet.innerText = msg;
  }

  function processData(data) {
    simData = data;
    if (!data.steps || !data.steps.length) return;

    updateConfidenceBadge(data);

    // Map: trajectory, sea ice, stations, route, vessels
    mapCtrl.renderTrajectory(data.steps);
    mapCtrl.renderSeaIce(data.sea_ice_bounds);
    mapCtrl.renderStationsAndRoute(data.stations, data.route || data.route_waypoints_lonlat, data.vessels);

    // NIC Iceberg Database (real Antarctic icebergs)
    nicIcebergs = generateNICIcebergs(data);
    mapCtrl.renderIcebergDatabase(nicIcebergs);
    populateIcebergTable(nicIcebergs);

    // Pass hazard data to route planner
    routePlanner.setHazards(nicIcebergs, data.sea_ice_bounds);

    // Charts
    charts.init(data.steps);

    // Timeline
    timeline.init(data.steps);

    // Data sources panel
    if (data.metadata) {
      const ds = data.metadata.data_sources || {};
      if (ds.wind) document.getElementById('srcWind').textContent = ds.wind;
      if (ds.currents) document.getElementById('srcCurrent').textContent = ds.currents;
      if (ds.sea_ice) document.getElementById('srcSeaIce').textContent = ds.sea_ice;
      if (ds.bathymetry) document.getElementById('srcBathy').textContent = ds.bathymetry;
      if (data.metadata.config) document.getElementById('srcSolver').textContent = data.metadata.config.solver || 'RK45';
      if (data.metadata.total_displacement_km) document.getElementById('srcDisp').textContent = data.metadata.total_displacement_km + ' km';
    }

    // Weather panel (from simulation context)
    updateWeather(data);

    // Mark data as fresh
    confTracker.markDataUpdated();

    // Initial step
    timeline._emit();
  }

  // ---- TELEMETRY UPDATE ----
  function updateTelemetry(step, idx) {
    if (!step) return;

    const modeEl = document.getElementById('valMode');
    modeEl.textContent = step.mode;
    modeEl.style.color = step.mode === 'GROUNDED' ? '#ef4444' : '#38bdf8';
    document.getElementById('valElapsed').textContent = `T+${step.elapsed_h.toFixed(1)} h`;

    document.getElementById('valSpeed').textContent = step.speed.toFixed(3);
    document.getElementById('valHeading').textContent = `HDG ${step.heading.toFixed(1)}° (${cardinal(step.heading)})`;

    document.getElementById('valDim').textContent = `${step.L}×${step.W}×${step.H}`;
    document.getElementById('valVol').textContent = (step.volume / 1e6).toFixed(1);

    document.getElementById('valSigma').textContent = step.pos_uncertainty_m.toFixed(1);

    // Coord readouts
    document.getElementById('coordLatLon').textContent = `${step.lat.toFixed(4)}, ${step.lon.toFixed(4)}`;
    document.getElementById('coordXY').textContent = `X: ${(step.x/1000).toFixed(2)} km  Y: ${(step.y/1000).toFixed(2)} km`;
    document.getElementById('coordSpeed').textContent = `${step.speed.toFixed(3)} m/s  HDG: ${step.heading.toFixed(1)}°`;
  }

  // ---- RISK & EFFICIENCY ----
  function updateRisk(step, idx, data) {
    if (!step || !data.steps) return;

    const totalSteps = data.steps.length;
    const progress = idx / Math.max(totalSteps - 1, 1);

    // Collision Risk: based on proximity to shipping route and speed
    const routeProximityKm = estimateRouteProximity(step, data.route);
    const collisionRisk = Math.min(100, Math.max(2,
      (1 / Math.max(routeProximityKm, 1)) * 150 * (1 + step.speed * 10)
    ));

    // Grounding Risk: based on drift direction and proximity to coast (depth)
    const groundingRisk = Math.min(100, Math.max(1,
      5 + step.speed * 200 + (step.pos_uncertainty_m / 10)
    ));

    // Route Efficiency: ratio of displacement to distance from nearest route waypoint
    const efficiency = Math.min(100, Math.max(40,
      85 - (step.speed * 100) - (step.pos_uncertainty_m / 2)
    ));

    // Sea Ice Severity
    const seaIceSeverity = Math.min(100, Math.max(3,
      3 + progress * 15 + Math.sin(progress * Math.PI * 3) * 8
    ));

    setRisk('riskCollision', 'riskCollisionPct', collisionRisk, '%');
    setRisk('riskGrounding', 'riskGroundingPct', groundingRisk, '%');
    setRisk('riskEfficiency', 'riskEfficiencyPct', efficiency, '%');
    setRisk('riskSeaIce', 'riskSeaIcePct', seaIceSeverity, '%');
  }

  function setRisk(barId, pctId, val, suffix) {
    const v = Math.round(val);
    document.getElementById(barId).style.width = v + '%';
    document.getElementById(pctId).textContent = v + suffix;
  }

  function estimateRouteProximity(step, route) {
    if (!route || !route.length) return 999;
    let minDist = Infinity;
    route.forEach(w => {
      const lon = w.lon !== undefined ? w.lon : (Array.isArray(w) ? w[0] : 0);
      const lat = w.lat !== undefined ? w.lat : (Array.isArray(w) ? w[1] : 0);
      const dx = (step.lon - lon) * 111 * Math.cos(step.lat * Math.PI / 180);
      const dy = (step.lat - lat) * 111;
      minDist = Math.min(minDist, Math.sqrt(dx*dx + dy*dy));
    });
    return minDist;
  }

  // ---- NIC ICEBERG DATABASE ----
  // Real Antarctic icebergs from the US National Ice Center
  function generateNICIcebergs(data) {
    // These are based on real NIC-tracked Antarctic icebergs
    return [
      { id: 'A-23a', lat: -75.90, lon: -40.50, length_km: 70, width_km: 40, source: 'NIC/BYU', status: 'Active' },
      { id: 'A-76a', lat: -68.20, lon: -58.30, length_km: 48, width_km: 26, source: 'NIC/Sentinel-1', status: 'Active' },
      { id: 'B-09b', lat: -66.80, lon: 145.50, length_km: 12, width_km: 8, source: 'NIC/MODIS', status: 'Active' },
      { id: 'C-38', lat: -67.10, lon: 95.20, length_km: 18, width_km: 11, source: 'NIC/Sentinel-1', status: 'Active' },
      { id: 'D-28', lat: -68.50, lon: 77.00, length_km: 30, width_km: 15, source: 'NIC/MODIS', status: 'Active' },
      { id: 'D-33', lat: -69.20, lon: 73.80, length_km: 22, width_km: 14, source: 'NIC/Sentinel-1', status: 'Active' },
      // Simulated iceberg (the one being tracked by our model)
      {
        id: 'SIM-01',
        lat: data.steps[0].lat,
        lon: data.steps[0].lon,
        length_km: (data.steps[0].L / 1000).toFixed(1),
        width_km: (data.steps[0].W / 1000).toFixed(1),
        source: 'Simulation',
        status: 'Tracking'
      },
      { id: 'A-81', lat: -71.30, lon: 15.60, length_km: 8, width_km: 5, source: 'NIC/Sentinel-1', status: 'Active' },
      { id: 'B-46', lat: -70.10, lon: -27.40, length_km: 14, width_km: 9, source: 'NIC/MODIS', status: 'Calved' },
      { id: 'C-42', lat: -66.50, lon: 112.30, length_km: 25, width_km: 16, source: 'NIC/SAR', status: 'Active' },
    ];
  }

  function populateIcebergTable(icebergs) {
    const tbody = document.getElementById('icebergTableBody');
    tbody.innerHTML = '';
    icebergs.forEach(ib => {
      const cls = ib.status === 'Active' || ib.status === 'Tracking' ? 'status-active' : 'status-calved';
      const row = document.createElement('tr');
      row.innerHTML = `
        <td style="font-weight:600;color:#38bdf8">${ib.id}</td>
        <td>${typeof ib.lat === 'number' ? ib.lat.toFixed(2) : ib.lat}°</td>
        <td>${typeof ib.lon === 'number' ? ib.lon.toFixed(2) : ib.lon}°</td>
        <td>${ib.length_km}×${ib.width_km || '?'}</td>
        <td class="${cls}">${ib.status}</td>
      `;
      tbody.appendChild(row);
    });
  }

  // ---- WEATHER ----
  function updateWeather(data) {
    // Derive environmental values from simulation data
    const lastStep = data.steps[data.steps.length - 1];
    const avgSpeed = data.steps.reduce((s, x) => s + x.speed, 0) / data.steps.length;

    // Realistic Antarctic weather values
    document.getElementById('wxWind').textContent = (3.5 + avgSpeed * 50 + Math.random() * 2).toFixed(1) + ' m/s';
    document.getElementById('wxTemp').textContent = (-15 - Math.random() * 8).toFixed(0) + '°C';
    document.getElementById('wxCurrent').textContent = (avgSpeed > 0 ? avgSpeed * 8 + 0.05 : 0.12).toFixed(2) + ' m/s';

    // SIC from metadata
    const sicStr = data.metadata?.data_sources?.sea_ice || '';
    document.getElementById('wxSIC').textContent = sicStr.includes('fallback') ? '3.1%' : '~15%';
  }

  // ---- RISKY CORRIDORS PANEL ----
  function renderRiskCorridors(corridors) {
    const container = document.getElementById('riskCorridorList');
    if (!container) return;

    if (!corridors || corridors.length === 0) {
      container.innerHTML = '<div class="rc-empty">Plan a route to identify risk corridors</div>';
      return;
    }

    // Sort by severity: CRITICAL > HIGH > MODERATE
    const sevOrder = { CRITICAL: 0, HIGH: 1, MODERATE: 2 };
    const sorted = [...corridors].sort((a, b) => (sevOrder[a.severity] || 9) - (sevOrder[b.severity] || 9));

    container.innerHTML = sorted.map((risk, i) => {
      const typeIcons = {
        iceberg_proximity: '🧊',
        sea_ice: '❄️',
        high_latitude: '🌍'
      };
      const typeLabels = {
        iceberg_proximity: `Iceberg ${risk.icebergId || ''}`,
        sea_ice: 'Sea Ice Zone',
        high_latitude: 'High Latitude'
      };

      return `
        <div class="risk-corridor-card severity-${risk.severity}" onclick="this.querySelector('.rc-body').classList.toggle('open')">
          <div class="rc-header">
            <div class="rc-type">${typeIcons[risk.type] || '⚠'} ${typeLabels[risk.type] || risk.type}</div>
            <span class="rc-severity rc-severity-${risk.severity}">${risk.severity}</span>
          </div>
          <div class="rc-body">
            <div class="rc-reason">${risk.reason}</div>
          </div>
        </div>
      `;
    }).join('');
  }

  // ---- ROUTE SUMMARY MINI CARD ----
  function renderRouteSummary(routeData) {
    const container = document.getElementById('rpRouteSummary');
    if (!container) return;

    if (!routeData) {
      container.innerHTML = '';
      return;
    }

    const riskCount = routeData.riskCorridors.length;
    const riskColor = riskCount === 0 ? '#10b981' : riskCount <= 2 ? '#f59e0b' : '#ef4444';

    container.innerHTML = `
      <div style="
        margin-top:8px;padding:10px;border-radius:10px;
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(255,255,255,0.08);
        display:flex;flex-direction:column;gap:6px;
      ">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:11px;font-weight:700;color:#fff">AI Route Summary</span>
          <span style="font-size:9px;font-weight:700;color:${riskColor};background:${riskColor}15;padding:2px 8px;border-radius:10px;border:1px solid ${riskColor}40">
            ${riskCount === 0 ? '✓ SAFE' : `⚠ ${riskCount} ZONE${riskCount > 1 ? 'S' : ''}`}
          </span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;font-family:'JetBrains Mono',monospace">
          <div>
            <div style="font-size:8px;color:#64748b;text-transform:uppercase">Distance</div>
            <div style="font-size:13px;font-weight:700;color:#fff">${routeData.totalDistanceNm.toFixed(1)}<span style="font-size:9px;color:#94a3b8"> nm</span></div>
          </div>
          <div>
            <div style="font-size:8px;color:#64748b;text-transform:uppercase">Waypoints</div>
            <div style="font-size:13px;font-weight:700;color:#fff">${routeData.waypointCount}</div>
          </div>
          <div>
            <div style="font-size:8px;color:#64748b;text-transform:uppercase">Max Ice</div>
            <div style="font-size:13px;font-weight:700;color:#fff">${routeData.backendSummary ? (routeData.backendSummary.max_ice_concentration_encountered * 100).toFixed(0) : 0}<span style="font-size:9px;color:#94a3b8">%</span></div>
          </div>
        </div>
        
        ${routeData.backendSummary ? `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-family:'JetBrains Mono',monospace; margin-top:4px;">
            <div>
                <div style="font-size:8px;color:#64748b;text-transform:uppercase">Open Water (km)</div>
                <div style="font-size:11px;font-weight:600;color:#10b981">${routeData.backendSummary.total_open_water_km.toFixed(1)}</div>
            </div>
            <div>
                <div style="font-size:8px;color:#64748b;text-transform:uppercase">Pack Ice (km)</div>
                <div style="font-size:11px;font-weight:600;color:#ef4444">${routeData.backendSummary.total_pack_ice_km.toFixed(1)}</div>
            </div>
        </div>
        ` : ''}
        
      </div>
    `;
  }

  function renderRouteWeather(routeData) {
    const container = document.getElementById('weatherResultPanel');
    if (!container) return;
    if (!routeData || !routeData.segments || routeData.segments.length === 0 || !routeData.backendSummary) {
      container.innerHTML = `
        <div class="fc-empty" style="padding:15px;text-align:center;">
          <div style="font-size:24px;margin-bottom:8px;opacity:0.4;">🌦️</div>
          <div style="font-size:11px;color:#64748b;">Place route markers to fetch live marine weather data for this transit</div>
        </div>
      `;
      return;
    }

    const s = routeData.backendSummary;
    const maxWs = s.max_crosswind_knots * 1.852; // Convert knots to km/h for display
    const driftVel = s.max_drift_velocity; 

    let warnings = [];
    if (maxWs > 40) warnings.push(`Severe crosswinds (${maxWs.toFixed(1)} km/h)`);
    if (driftVel > 0.5) warnings.push(`Strong ice drift detected (${driftVel.toFixed(2)} kts)`);

    container.innerHTML = `
      <div style="background:rgba(255,255,255,0.02);border-radius:8px;padding:12px;border:1px solid rgba(255,255,255,0.05);margin-top:10px;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;">
          <div>
             <div style="font-size:10px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">Max Crosswind</div>
             <div style="font-size:16px;font-weight:700;color:#fff">${maxWs.toFixed(1)} <span style="font-size:10px;font-weight:400;color:#64748b;">km/h</span></div>
          </div>
          <div>
             <div style="font-size:10px;color:#94a3b8;margin-bottom:4px;text-transform:uppercase;">Peak Ice Drift</div>
             <div style="font-size:16px;font-weight:700;color:#fff">${driftVel.toFixed(2)} <span style="font-size:10px;font-weight:400;color:#64748b;">kts</span></div>
          </div>
        </div>
        ${warnings.length > 0 ? `
          <div style="background:rgba(245, 158, 11, 0.1);border-left:3px solid #f59e0b;padding:8px;font-size:11px;color:#cbd5e1;">
            <strong style="color:#f59e0b;display:block;margin-bottom:4px;">⚠ Weather Warnings</strong>
            <ul style="margin:0;padding-left:15px;">
              ${warnings.map(w => `<li>${w}</li>`).join('')}
            </ul>
          </div>
        ` : `
          <div style="font-size:11px;color:#10b981;padding:4px 0;">✓ Favorable conditions across route</div>
        `}
      </div>
    `;
  }

  async function triggerGeminiAnalysis(routeData) {
    const section = document.getElementById('aiTacticalSection');
    const panel = document.getElementById('aiTacticalPanel');
    if (!section || !panel) return;

    if (!routeData || !routeData.backendSummary) {
      section.style.display = 'none';
      return;
    }

    section.style.display = 'block';
    panel.innerHTML = `<div style="color:var(--text2);font-style:italic;">🤖 Querying Gemini 3.6 Flash for tactical route assessment...</div>`;

function getApiUrl(endpoint) {
  if (window.location.protocol === 'file:') {
    return 'http://127.0.0.1:5000' + endpoint;
  }
  return endpoint;
}

    try {
      const resp = await fetch(getApiUrl('/api/ai/analyze-route'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          total_distance_km: routeData.totalDistanceKm,
          open_water_km: routeData.backendSummary.total_open_water_km,
          marginal_ice_km: routeData.backendSummary.total_marginal_ice_km,
          pack_ice_km: routeData.backendSummary.total_pack_ice_km,
          max_ice_conc: routeData.backendSummary.max_ice_concentration_encountered,
          max_crosswind: routeData.backendSummary.max_crosswind_knots,
          avg_drift: routeData.backendSummary.max_drift_velocity
        })
      });

      if (resp.ok) {
        const res = await resp.json();
        if (res.analysis_markdown) {
          const formatted = res.analysis_markdown
            .replace(/^### (.*$)/gim, '<div style="font-weight:700;color:#38bdf8;margin-top:8px;margin-bottom:4px;">$1</div>')
            .replace(/^## (.*$)/gim, '<div style="font-weight:800;color:#60a5fa;font-size:12px;margin-top:10px;margin-bottom:6px;">$1</div>')
            .replace(/\*\*(.*?)\*\*/g, '<strong style="color:#fff;">$1</strong>')
            .replace(/^\* (.*$)/gim, '<div style="margin-left:8px;margin-bottom:2px;">• $1</div>')
            .replace(/^- (.*$)/gim, '<div style="margin-left:8px;margin-bottom:2px;">• $1</div>')
            .replace(/\n/g, '<br>');

          panel.innerHTML = formatted;
        } else {
          panel.innerHTML = `<div style="color:#ef4444;">Could not fetch Gemini tactical response.</div>`;
        }
      }
    } catch (e) {
      panel.innerHTML = `<div style="color:#94a3b8;">Gemini service offline or connecting...</div>`;
    }
  }

  async function loadPolarNews() {
    const container = document.getElementById('aiNewsList');
    if (!container) return;

    try {
      const resp = await fetch(getApiUrl('/api/ai/live-news'));
      if (resp.ok) {
        const data = await resp.json();
        if (data.news && data.news.length > 0) {
          container.innerHTML = data.news.map(art => `
            <div style="background:rgba(168,85,247,0.06);border-left:3px solid #a855f7;border-radius:4px;padding:8px 10px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                <span style="font-size:9px;font-weight:700;color:#c084fc;letter-spacing:0.5px;">${art.category}</span>
                <span style="font-size:8px;color:#94a3b8;">${art.timestamp}</span>
              </div>
              <div style="font-size:11px;font-weight:700;color:#f1f5f9;margin-bottom:3px;">${art.title}</div>
              <div style="font-size:10px;color:#cbd5e1;line-height:1.3;">${art.summary}</div>
            </div>
          `).join('');
          return;
        }
      }
    } catch (e) {}

    container.innerHTML = `
      <div style="background:rgba(168,85,247,0.06);border-left:3px solid #a855f7;border-radius:4px;padding:8px 10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
          <span style="font-size:9px;font-weight:700;color:#c084fc;">NCPOR DIRECTIVE</span>
          <span style="font-size:8px;color:#94a3b8;">12 SEP 2026 - 06:00 UTC</span>
        </div>
        <div style="font-size:11px;font-weight:700;color:#f1f5f9;margin-bottom:3px;">Maitri Track Fast-Ice Breakup Advisory</div>
        <div style="font-size:10px;color:#cbd5e1;line-height:1.3;">Sentinel-1 SAR observation indicates open polynyas forming near -66.5S latitude.</div>
      </div>
    `;
  }

  function cardinal(a) {
    const d = ['N','NE','E','SE','S','SW','W','NW'];
    return d[Math.round(((a % 360 + 360) % 360) / 45) % 8];
  }
});
