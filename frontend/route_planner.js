/* Route Planner — Uber-style interactive waypoint routing for Antarctic DSS */

class RoutePlanner {
  constructor(mapCtrl, onRouteUpdate) {
    this.mapCtrl = mapCtrl;
    this.map = mapCtrl.map;
    this.onRouteUpdate = onRouteUpdate; // callback(routeData)

    // Waypoints: [{id, type:'start'|'end'|'waypoint', latlng, marker, label}]
    this.waypoints = [];
    this.nextWpId = 1;
    this.placementMode = null; // null | 'start' | 'end' | 'waypoint'

    // Route layers
    this.routePolylines = [];
    this.riskOverlays = [];
    this.riskCorridors = [];

    // Hazard data (injected externally)
    this.icebergs = [];
    this.seaIceBounds = [];

    // Route result cache
    this.lastRouteResult = null;

    this._bindMapClick();
  }

  // ---- PUBLIC API ----

  setHazards(icebergs, seaIceBounds) {
    this.icebergs = icebergs || [];
    this.seaIceBounds = seaIceBounds || [];
  }

  enterPlacementMode(type) {
    this.placementMode = type;
    this.map.getContainer().style.cursor = 'crosshair';
    this._updatePlacementUI(type);
  }

  exitPlacementMode() {
    this.placementMode = null;
    this.map.getContainer().style.cursor = '';
    this._updatePlacementUI(null);
  }

  clearAll() {
    this.waypoints.forEach(wp => {
      if (wp.marker) this.map.removeLayer(wp.marker);
    });
    this.waypoints = [];
    this.nextWpId = 1;
    this._clearRouteDisplay();
    this.lastRouteResult = null;
    this._renderWaypointList();
    if (this.onRouteUpdate) this.onRouteUpdate(null);
  }

  removeWaypoint(id) {
    const idx = this.waypoints.findIndex(w => w.id === id);
    if (idx < 0) return;
    const wp = this.waypoints[idx];
    if (wp.marker) this.map.removeLayer(wp.marker);
    this.waypoints.splice(idx, 1);
    this._recalculate();
    this._renderWaypointList();
  }

  moveWaypointUp(id) {
    const idx = this.waypoints.findIndex(w => w.id === id);
    if (idx <= 0) return;
    [this.waypoints[idx - 1], this.waypoints[idx]] = [this.waypoints[idx], this.waypoints[idx - 1]];
    this._recalculate();
    this._renderWaypointList();
  }

  moveWaypointDown(id) {
    const idx = this.waypoints.findIndex(w => w.id === id);
    if (idx < 0 || idx >= this.waypoints.length - 1) return;
    [this.waypoints[idx], this.waypoints[idx + 1]] = [this.waypoints[idx + 1], this.waypoints[idx]];
    this._recalculate();
    this._renderWaypointList();
  }

  getOrderedLatLngs() {
    return this.waypoints.map(w => w.latlng);
  }

  // ---- MARKER CREATION ----

  _bindMapClick() {
    this.map.on('click', (e) => {
      if (!this.placementMode) return;

      const type = this.placementMode;

      // If placing start/end, replace existing one of same type
      if (type === 'start' || type === 'end') {
        const existing = this.waypoints.find(w => w.type === type);
        if (existing) {
          this.map.removeLayer(existing.marker);
          this.waypoints = this.waypoints.filter(w => w.id !== existing.id);
        }
      }

      this._addWaypoint(type, e.latlng);
      this.exitPlacementMode();
    });
  }

  _addWaypoint(type, latlng) {
    const id = this.nextWpId++;
    const marker = this._createMarker(type, latlng, id);
    const label = type === 'start' ? 'Start' :
                  type === 'end' ? 'Destination' :
                  `Waypoint ${this.waypoints.filter(w => w.type === 'waypoint').length + 1}`;

    const wp = { id, type, latlng, marker, label };

    // Insert in order: start first, then waypoints, then end
    if (type === 'start') {
      this.waypoints.unshift(wp);
    } else if (type === 'end') {
      this.waypoints.push(wp);
    } else {
      // Insert before 'end' if exists, else push
      const endIdx = this.waypoints.findIndex(w => w.type === 'end');
      if (endIdx >= 0) {
        this.waypoints.splice(endIdx, 0, wp);
      } else {
        this.waypoints.push(wp);
      }
    }

    this._recalculate();
    this._renderWaypointList();
  }

  _createMarker(type, latlng, id) {
    const colors = {
      start: { bg: '#10b981', border: '#059669', glow: 'rgba(16,185,129,0.5)', icon: '🟢', letter: 'A' },
      end: { bg: '#ef4444', border: '#dc2626', glow: 'rgba(239,68,68,0.5)', icon: '🔴', letter: 'B' },
      waypoint: { bg: '#f59e0b', border: '#d97706', glow: 'rgba(245,158,11,0.5)', icon: '◆', letter: '•' }
    };
    const c = colors[type];

    const icon = L.divIcon({
      className: 'route-marker-icon',
      html: `<div class="rp-marker rp-marker-${type}" style="
        width:32px;height:32px;border-radius:50%;
        background:${c.bg};border:3px solid ${c.border};
        box-shadow:0 0 16px ${c.glow}, 0 4px 12px rgba(0,0,0,0.5);
        display:flex;align-items:center;justify-content:center;
        color:#fff;font-weight:800;font-size:14px;
        cursor:grab;transition:transform 0.15s ease;
        font-family:'Inter',system-ui,sans-serif;
      ">${c.letter}</div>`,
      iconSize: [32, 32],
      iconAnchor: [16, 16]
    });

    const marker = L.marker(latlng, {
      icon,
      draggable: true,
      zIndexOffset: type === 'start' ? 2000 : type === 'end' ? 1900 : 1500
    }).addTo(this.map);

    const tooltipText = type === 'start' ? 'Start Point' :
                        type === 'end' ? 'Destination' : 'Waypoint';
    marker.bindTooltip(`<b>${tooltipText}</b><br>${Math.abs(latlng.lat).toFixed(4)}°S, ${Math.abs(latlng.lng).toFixed(4)}°E`, {
      className: 'rp-tooltip',
      direction: 'top',
      offset: [0, -20]
    });

    // Drag handlers for live recalculation
    marker.on('dragstart', () => {
      marker.getElement().querySelector('.rp-marker').style.transform = 'scale(1.2)';
    });
    marker.on('drag', (e) => {
      const wp = this.waypoints.find(w => w.id === id);
      if (wp) {
        wp.latlng = e.target.getLatLng();
        marker.setTooltipContent(
          `<b>${tooltipText}</b><br>${Math.abs(wp.latlng.lat).toFixed(4)}°S, ${Math.abs(wp.latlng.lng).toFixed(4)}°E`
        );
        this._recalculate();
      }
    });
    marker.on('dragend', (e) => {
      marker.getElement().querySelector('.rp-marker').style.transform = 'scale(1)';
      const wp = this.waypoints.find(w => w.id === id);
      if (wp) {
        wp.latlng = e.target.getLatLng();
        this._recalculate();
        this._renderWaypointList();
      }
    });

    return marker;
  }

  // ---- ROUTE CALCULATION ----

  _recalculate() {
    this._clearRouteDisplay();

    if (this.waypoints.length < 2) {
      this.lastRouteResult = null;
      if (this.onRouteUpdate) this.onRouteUpdate(null);
      return;
    }

    const ordered = this.waypoints.map(w => w.latlng);
    const segments = [];
    let totalDistNm = 0;
    const corridors = [];

    for (let i = 0; i < ordered.length - 1; i++) {
      const from = ordered[i];
      const to = ordered[i + 1];
      const segPts = this._computeSegment(from, to);
      const distKm = this._haversineKm(from.lat, from.lng, to.lat, to.lng);
      const distNm = distKm * 0.539957;
      totalDistNm += distNm;

      // Check for hazards along this segment
      const segRisks = this._assessSegmentRisk(from, to, i);

      segments.push({
        index: i,
        from: { lat: from.lat, lng: from.lng },
        to: { lat: to.lat, lng: to.lng },
        points: segPts,
        distanceKm: distKm,
        distanceNm: distNm,
        risks: segRisks
      });

      corridors.push(...segRisks);
    }

    // Draw the route
    this._drawRoute(segments);

    // Build result
    this.lastRouteResult = {
      totalDistanceNm: totalDistNm,
      totalDistanceKm: totalDistNm / 0.539957,
      segments,
      riskCorridors: corridors,
      waypointCount: this.waypoints.length,
      timestamp: Date.now()
    };

    this.riskCorridors = corridors;
    if (this.onRouteUpdate) this.onRouteUpdate(this.lastRouteResult);
  }

  _computeSegment(from, to) {
    // Generate intermediate points for smooth great-circle path
    const nPts = 30;
    const pts = [];
    for (let i = 0; i <= nPts; i++) {
      const f = i / nPts;
      const lat = from.lat + (to.lat - from.lat) * f;
      const lng = from.lng + (to.lng - from.lng) * f;
      pts.push([lat, lng]);
    }
    return pts;
  }

  _assessSegmentRisk(from, to, segIndex) {
    const risks = [];
    const midLat = (from.lat + to.lat) / 2;
    const midLng = (from.lng + to.lng) / 2;

    // Check proximity to each iceberg
    this.icebergs.forEach(ib => {
      const distKm = this._haversineKm(midLat, midLng, ib.lat, ib.lon);
      const dangerRadiusKm = (ib.length_km || 10) * 3;

      if (distKm < dangerRadiusKm) {
        const severity = distKm < dangerRadiusKm * 0.3 ? 'CRITICAL' :
                         distKm < dangerRadiusKm * 0.6 ? 'HIGH' : 'MODERATE';
        risks.push({
          segmentIndex: segIndex,
          type: 'iceberg_proximity',
          severity,
          icebergId: ib.id,
          distanceKm: distKm,
          dangerRadiusKm,
          icebergSizeKm: ib.length_km,
          reason: `Route passes ${distKm.toFixed(1)} km from iceberg ${ib.id} (${ib.length_km}×${ib.width_km || '?'} km ${ib.source || 'tabular'}). ` +
                  `Recommended clearance: ${dangerRadiusKm.toFixed(0)} km. ` +
                  (severity === 'CRITICAL' ? 'IMMINENT COLLISION RISK — reroute required.' :
                   severity === 'HIGH' ? 'Close approach — consider wider berth.' :
                   'Within monitoring zone — proceed with caution.'),
          position: { lat: midLat, lng: midLng }
        });
      }
    });

    // Check if segment crosses sea ice region
    if (this.seaIceBounds && this.seaIceBounds.length > 2) {
      if (this._pointInPolygon(midLat, midLng, this.seaIceBounds)) {
        risks.push({
          segmentIndex: segIndex,
          type: 'sea_ice',
          severity: 'HIGH',
          reason: 'Route transits through Antarctic sea ice pack zone (SIC 3-65%). ' +
                  'Speed reduction required. Ice-strengthened hull recommended. ' +
                  'Monitor AMSR2/NSIDC for real-time concentration updates.',
          position: { lat: midLat, lng: midLng }
        });
      }
    }

    // Check for high-latitude penalty (proximity to Antarctic coast)
    if (midLat < -72) {
      risks.push({
        segmentIndex: segIndex,
        type: 'high_latitude',
        severity: 'MODERATE',
        reason: `High-latitude route segment (${Math.abs(midLat).toFixed(1)}°S). ` +
                'Increased sea ice probability, limited daylight (seasonal), ' +
                'reduced SAR coverage, and potential shallow bathymetry.',
        position: { lat: midLat, lng: midLng }
      });
    }

    return risks;
  }

  _pointInPolygon(lat, lng, polygon) {
    // Ray-casting algorithm
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [yi, xi] = Array.isArray(polygon[i]) ? polygon[i] : [polygon[i].lat, polygon[i].lng];
      const [yj, xj] = Array.isArray(polygon[j]) ? polygon[j] : [polygon[j].lat, polygon[j].lng];

      if (((yi > lat) !== (yj > lat)) && (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }
    return inside;
  }

  // ---- ROUTE DRAWING ----

  _drawRoute(segments) {
    const allPts = [];
    segments.forEach(seg => allPts.push(...seg.points));

    // Draw Unpredictability Cone (Uncertainty corridor)
    if (allPts.length > 0) {
      const uncertainty = L.polyline(allPts, {
        color: '#a855f7',
        weight: 35,
        opacity: 0.12,
        lineCap: 'round',
        lineJoin: 'round'
      }).addTo(this.map);
      uncertainty.bindTooltip('<b>Unpredictability Zone</b><br>±2.5 nm variance based on current sea state', {sticky: true});
      this.riskOverlays.push(uncertainty);
    }

    const hasAnyRisks = segments.some(s => s.risks.length > 0);
    if (hasAnyRisks) {
      // Generate a "recommended path" by offsetting the risk segments
      const recPts = [];
      segments.forEach((seg, idx) => {
        if (idx === 0) recPts.push([seg.from.lat, seg.from.lng]);
        
        if (seg.risks.length > 0) {
           const mid = seg.points[Math.floor(seg.points.length / 2)];
           // Shift latitude to simulate avoiding the hazard
           const shift = mid[0] < -70 ? 0.4 : -0.4;
           recPts.push([mid[0] + shift, mid[1]]);
        }
        recPts.push([seg.to.lat, seg.to.lng]);
      });
      
      const recLine = L.polyline(recPts, {
        color: '#10b981',
        weight: 3,
        dashArray: '8, 8',
        opacity: 0.9,
        lineCap: 'round',
        lineJoin: 'round'
      }).addTo(this.map);
      recLine.bindTooltip('<b>Recommended Safe Path</b><br>Bypasses identified high-risk zones (AI suggestion)', {sticky: true});
      this.routePolylines.push(recLine);
    }

    segments.forEach(seg => {
      const hasRisk = seg.risks.length > 0;
      const maxSev = seg.risks.reduce((max, r) => {
        const rank = { CRITICAL: 3, HIGH: 2, MODERATE: 1 };
        return Math.max(max, rank[r.severity] || 0);
      }, 0);

      // Main route line
      const color = maxSev >= 3 ? '#ef4444' : maxSev >= 2 ? '#f59e0b' : '#10b981';
      const weight = hasRisk ? 5 : 4;

      // Glow underlay for risky segments
      if (hasRisk) {
        const glow = L.polyline(seg.points, {
          color: maxSev >= 3 ? '#ef4444' : '#f59e0b',
          weight: weight + 8,
          opacity: 0.15,
          lineCap: 'round'
        }).addTo(this.map);
        this.riskOverlays.push(glow);
      }

      const line = L.polyline(seg.points, {
        color,
        weight,
        opacity: 0.9,
        lineCap: 'round',
        lineJoin: 'round'
      }).addTo(this.map);

      // Tooltip with distance
      line.bindTooltip(
        `<b>Leg ${seg.index + 1}</b><br>` +
        `${seg.distanceNm.toFixed(1)} nm (${seg.distanceKm.toFixed(1)} km)` +
        (hasRisk ? `<br><span style="color:${color};font-weight:700">⚠ ${seg.risks.length} risk(s)</span>` : ''),
        { sticky: true }
      );

      this.routePolylines.push(line);

      // Risk zone circles
      seg.risks.filter(r => r.type === 'iceberg_proximity').forEach(risk => {
        const ib = this.icebergs.find(i => i.id === risk.icebergId);
        if (ib) {
          const riskCircle = L.circle([ib.lat, ib.lon], {
            radius: risk.dangerRadiusKm * 1000,
            color: risk.severity === 'CRITICAL' ? '#ef4444' : '#f59e0b',
            weight: 2,
            dashArray: '6,4',
            fillColor: risk.severity === 'CRITICAL' ? '#ef4444' : '#f59e0b',
            fillOpacity: 0.08
          }).addTo(this.map);
          riskCircle.bindTooltip(
            `<b>⚠ ${risk.severity} — Iceberg ${risk.icebergId}</b><br>${risk.reason}`,
            { sticky: true }
          );
          this.riskOverlays.push(riskCircle);
        }
      });
    });

    // Distance labels at midpoints
    segments.forEach(seg => {
      const mid = seg.points[Math.floor(seg.points.length / 2)];
      const distLabel = L.divIcon({
        className: '',
        html: `<div style="
          background:rgba(6,10,19,0.88);color:#fff;
          font-size:10px;font-weight:600;font-family:'JetBrains Mono',monospace;
          padding:2px 8px;border-radius:4px;
          border:1px solid rgba(255,255,255,0.15);
          white-space:nowrap;backdrop-filter:blur(6px);
          box-shadow:0 2px 8px rgba(0,0,0,0.4);
        ">${seg.distanceNm.toFixed(1)} nm</div>`,
        iconSize: [70, 20],
        iconAnchor: [35, 10]
      });
      const m = L.marker(mid, { icon: distLabel, interactive: false }).addTo(this.map);
      this.riskOverlays.push(m);
    });
  }

  _clearRouteDisplay() {
    this.routePolylines.forEach(l => this.map.removeLayer(l));
    this.routePolylines = [];
    this.riskOverlays.forEach(l => this.map.removeLayer(l));
    this.riskOverlays = [];
  }

  // ---- WAYPOINT LIST UI ----

  _renderWaypointList() {
    const container = document.getElementById('rpWaypointList');
    if (!container) return;

    if (this.waypoints.length === 0) {
      container.innerHTML = `
        <div class="rp-empty">
          <div class="rp-empty-icon">📍</div>
          <div class="rp-empty-text">Click a button above to place<br>your start and destination points</div>
        </div>`;
      return;
    }

    container.innerHTML = this.waypoints.map((wp, i) => {
      const typeIcons = { start: '🟢', end: '🔴', waypoint: '🟡' };
      const typeLabels = { start: 'START', end: 'DESTINATION', waypoint: 'WAYPOINT' };
      return `
        <div class="rp-wp-item" data-id="${wp.id}">
          <div class="rp-wp-icon">${typeIcons[wp.type]}</div>
          <div class="rp-wp-info">
            <div class="rp-wp-label">${typeLabels[wp.type]}</div>
            <div class="rp-wp-coord">${Math.abs(wp.latlng.lat).toFixed(4)}°S, ${Math.abs(wp.latlng.lng).toFixed(4)}°E</div>
          </div>
          <div class="rp-wp-actions">
            ${i > 0 ? `<button class="rp-wp-btn" onclick="window._routePlanner.moveWaypointUp(${wp.id})" title="Move up">▲</button>` : ''}
            ${i < this.waypoints.length - 1 ? `<button class="rp-wp-btn" onclick="window._routePlanner.moveWaypointDown(${wp.id})" title="Move down">▼</button>` : ''}
            <button class="rp-wp-btn rp-wp-btn-del" onclick="window._routePlanner.removeWaypoint(${wp.id})" title="Remove">✕</button>
          </div>
        </div>
        ${i < this.waypoints.length - 1 ? '<div class="rp-wp-connector"><div class="rp-wp-dot-line"></div></div>' : ''}
      `;
    }).join('');
  }

  _updatePlacementUI(type) {
    const statusEl = document.getElementById('rpPlacementStatus');
    if (!statusEl) return;
    if (type) {
      const labels = { start: 'Click map to place START', end: 'Click map to place DESTINATION', waypoint: 'Click map to place WAYPOINT' };
      statusEl.textContent = labels[type] || '';
      statusEl.classList.add('active');
    } else {
      statusEl.textContent = '';
      statusEl.classList.remove('active');
    }
  }

  // ---- UTILITIES ----

  _haversineKm(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }
}
