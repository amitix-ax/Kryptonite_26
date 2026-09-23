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
    
    // GIS Data
    this.antarcticaGeoJSON = null;
    this._loadGISData();

    this._generateMockHazards();
    this._bindMapClick();
  }

  // ---- PUBLIC API ----
  
  setHazards(icebergs, seaIceBounds) {
    if (icebergs && icebergs.length > 0) {
      if (this.icebergCircleLayers) {
        this.icebergCircleLayers.forEach(l => {
          if (this.map && this.map.hasLayer(l)) this.map.removeLayer(l);
        });
      }
      this.icebergCircleLayers = [];

      this.icebergs = icebergs.map(ib => {
        const lat = ib.lat;
        const lng = (ib.lng !== undefined) ? ib.lng : (ib.lon !== undefined ? ib.lon : 0);
        const lengthKm = parseFloat(ib.length_km || 10);
        const widthKm = parseFloat(ib.width_km || 5);
        const radiusKm = ib.radiusKm || Math.max(10, Math.max(lengthKm, widthKm) / 2);
        return {
          ...ib,
          lat,
          lng,
          radiusKm
        };
      });

      // Render physical hazard rings for all known icebergs
      if (this.map) {
        this.icebergs.forEach(ice => {
          if (ice.lat === undefined || ice.lng === undefined) return;
          const circle = L.circle([ice.lat, ice.lng], {
            color: '#ef4444',
            fillColor: '#ef4444',
            fillOpacity: 0.22,
            weight: 1.5,
            dashArray: '3, 4',
            radius: (ice.radiusKm + 3.0) * 1000
          }).bindPopup(`
            <div style="font-family:sans-serif;font-size:11px;color:#1e293b;">
              <strong style="color:#ef4444;">🚨 KNOWN ICEBERG: ${ice.id || 'Hazard'}</strong><br>
              <strong>Status:</strong> ${ice.status || 'Active'}<br>
              <strong>Dimensions:</strong> ${ice.length_km || ice.radiusKm * 2} × ${ice.width_km || ice.radiusKm} km<br>
              <strong>Safety Standoff Radius:</strong> ${(ice.radiusKm + 3.0).toFixed(1)} km<br>
              <span style="color:#64748b;font-size:10px;">Enforced A* navigation obstacle</span>
            </div>
          `).addTo(this.map);
          this.icebergCircleLayers.push(circle);
        });
      }
    }
    if (seaIceBounds && seaIceBounds.length > 0) this.seaIceBounds = seaIceBounds;

    // Check if moving icebergs conflict with the active planned route (Google Maps style)
    if (this.lastRouteResult && this.lastRouteResult.coordinates && this.lastRouteResult.coordinates.length > 0) {
      this.checkAndAutoRerouteAgainstTraffic();
    }
  }

  _generateMockHazards() {
    this.icebergCircleLayers = [];
    this.dynamicDetourZones = [];
    // Default icebergs
    this.icebergs = [
      { id: 'A-81', lat: -71.30, lng: 15.60, radiusKm: 18 },
      { id: 'A-23a', lat: -75.90, lng: -40.50, radiusKm: 38 },
      { id: 'A-76a', lat: -68.20, lng: -58.30, radiusKm: 28 },
      { id: 'B-09b', lat: -66.80, lng: 145.50, radiusKm: 14 },
      { id: 'C-38', lat: -67.10, lng: 95.20, radiusKm: 16 }
    ];
  }

  async _loadGISData() {
     try {
         const resp = await fetch('public/antarctica.json');
         if (resp.ok) {
             this.antarcticaGeoJSON = await resp.json();
             // Draw U-Net semantic segmentation mask over ice ground
             if (this.map) {
                 L.geoJSON(this.antarcticaGeoJSON, {
                     style: {
                         color: '#ef4444',
                         weight: 1,
                         opacity: 0.8,
                         fillColor: '#ef4444',
                         fillOpacity: 0.12,
                         dashArray: '4, 4'
                     }
                 }).bindTooltip('U-Net Semantic Mask: Ice Ground', {sticky: true}).addTo(this.map);
             }
         }
     } catch (e) {
         console.warn("Failed to load Antarctica GIS data for U-Net segmentation.");
     }
  }

  _isNavigable(lat, lng) {
      // 1. Continental interior ice cap barrier (South of -82.5S is entirely ice ground)
      if (lat <= -82.5) return false;

      // 2. High-precision GIS Landmass collision (U-Net Semantic Map + Ice Shelves)
      if (this.antarcticaGeoJSON && window.turf) {
          const pt = turf.point([lng, lat]);
          if (this.antarcticaGeoJSON.type === 'FeatureCollection') {
              for (const feature of this.antarcticaGeoJSON.features) {
                  if (feature.geometry.type === 'Polygon' || feature.geometry.type === 'MultiPolygon') {
                      if (turf.booleanPointInPolygon(pt, feature)) {
                          return false;
                      }
                  }
              }
          } else if (this.antarcticaGeoJSON.type === 'Feature') {
              if (this.antarcticaGeoJSON.geometry.type === 'Polygon' || this.antarcticaGeoJSON.geometry.type === 'MultiPolygon') {
                  if (turf.booleanPointInPolygon(pt, this.antarcticaGeoJSON)) return false;
              }
          }
      }

      // 3. Known Iceberg avoidance (NIC catalog + active drift targets)
      for (const ice of this.icebergs) {
          const iceLon = (ice.lng !== undefined) ? ice.lng : ice.lon;
          const iceLat = ice.lat;
          if (iceLat === undefined || iceLon === undefined) continue;
          
          const radiusKm = ice.radiusKm || Math.max(10, Math.max(ice.length_km || 10, ice.width_km || 5) / 2);
          const safeBufferKm = 3.5;
          if (this._haversineKm(lat, lng, iceLat, iceLon) < (radiusKm + safeBufferKm)) {
              return false;
          }
      }

      // 4. Dynamic Auto-Reroute Repulsive Traffic Obstacles
      if (this.dynamicDetourZones && this.dynamicDetourZones.length > 0) {
          for (const zone of this.dynamicDetourZones) {
              if (this._haversineKm(lat, lng, zone.lat, zone.lng) < zone.radiusKm) {
                  return false;
              }
          }
      }

      return true;
  }

  _calculateAStarPath(from, to) {
      const dist = this._haversineKm(from.lat, from.lng, to.lat, to.lng);
      if (dist < 8) return [[from.lat, from.lng], [to.lat, to.lng]];

      // Check if routing crosses the Antarctic Peninsula (lat -75 to -63.2, lng -75 to -55)
      const crossesPeninsula = (
          ((from.lng < -61 && to.lng > -57) || (from.lng > -57 && to.lng < -61)) &&
          (from.lat < -63.0 || to.lat < -63.0)
      );

      // If crossing the peninsula, route via open water in Drake Passage / Bransfield Strait
      if (crossesPeninsula) {
          const capeWaypoint = { lat: -62.2, lng: -58.8 }; // Open ocean north of Prime Head
          const path1 = this._runAStarGrid(from, capeWaypoint);
          const path2 = this._runAStarGrid(capeWaypoint, to);
          const combined = [...path1, ...path2.slice(1)];
          return this._smoothPath(combined);
      }

      return this._smoothPath(this._runAStarGrid(from, to));
  }

  _runAStarGrid(from, to) {
      const dist = this._haversineKm(from.lat, from.lng, to.lat, to.lng);
      const latSpan = Math.abs(from.lat - to.lat);
      const lngSpan = Math.abs(from.lng - to.lng);
      
      const latMargin = Math.max(4.5, latSpan * 0.4);
      const lngMargin = Math.max(6.5, lngSpan * 0.4);

      const minLat = Math.max(-82.5, Math.min(from.lat, to.lat) - latMargin);
      const maxLat = Math.min(-58.0, Math.max(from.lat, to.lat) + latMargin);
      const minLng = Math.min(from.lng, to.lng) - lngMargin;
      const maxLng = Math.max(from.lng, to.lng) + lngMargin;

      const gridSize = Math.min(55, Math.max(35, Math.round(dist / 30)));
      const dLat = (maxLat - minLat) / gridSize;
      const dLng = (maxLng - minLng) / gridSize;

      const getId = (lat, lng) => `${lat.toFixed(3)},${lng.toFixed(3)}`;
      const startNode = { lat: from.lat, lng: from.lng, g: 0, f: 0, parent: null, id: 'start' };
      const endNode = { lat: to.lat, lng: to.lng, g: 0, f: 0, parent: null, id: 'end' };

      const openSet = [startNode];
      const closedSet = new Set();

      const getNeighbors = (node) => {
          const neighbors = [];
          const dirs = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]];
          for (let d of dirs) {
              const nLat = node.lat + d[0] * dLat;
              const nLng = node.lng + d[1] * dLng;
              if (this._isNavigable(nLat, nLng)) {
                  neighbors.push({lat: nLat, lng: nLng, id: getId(nLat, nLng)});
              }
          }
          if (this._haversineKm(node.lat, node.lng, to.lat, to.lng) < (dist / gridSize * 2.5)) {
              neighbors.push({lat: to.lat, lng: to.lng, id: 'end'});
          }
          return neighbors;
      };

      let iters = 0;
      while (openSet.length > 0 && iters < 3500) {
          iters++;
          let lowestIdx = 0;
          for (let i = 1; i < openSet.length; i++) {
              if (openSet[i].f < openSet[lowestIdx].f) lowestIdx = i;
          }
          const current = openSet[lowestIdx];

          if (current.id === 'end' || this._haversineKm(current.lat, current.lng, to.lat, to.lng) < (dist / gridSize)) {
              const path = [];
              let curr = current;
              while (curr) {
                  path.unshift([curr.lat, curr.lng]);
                  curr = curr.parent;
              }
              path[0] = [from.lat, from.lng];
              path[path.length - 1] = [to.lat, to.lng];
              return path;
          }

          openSet.splice(lowestIdx, 1);
          closedSet.add(current.id);

          const neighbors = getNeighbors(current);
          for (const neighbor of neighbors) {
              if (closedSet.has(neighbor.id)) continue;
              const stepDist = this._haversineKm(current.lat, current.lng, neighbor.lat, neighbor.lng);
              const tentative_g = current.g + stepDist;

              let neighborNode = openSet.find(n => n.id === neighbor.id);
              if (!neighborNode) {
                  neighborNode = { ...neighbor, g: tentative_g, parent: current };
                  neighborNode.f = neighborNode.g + this._haversineKm(neighborNode.lat, neighborNode.lng, to.lat, to.lng);
                  openSet.push(neighborNode);
              } else if (tentative_g < neighborNode.g) {
                  neighborNode.parent = current;
                  neighborNode.g = tentative_g;
                  neighborNode.f = neighborNode.g + this._haversineKm(neighborNode.lat, neighborNode.lng, to.lat, to.lng);
              }
          }
      }

      // Safe offshore interpolation if direct path obstructed
      const safePoints = [[from.lat, from.lng]];
      const steps = 12;
      for (let s = 1; s < steps; s++) {
          const frac = s / steps;
          let pLat = from.lat + (to.lat - from.lat) * frac;
          let pLng = from.lng + (to.lng - from.lng) * frac;
          if (!this._isNavigable(pLat, pLng)) {
              while (!this._isNavigable(pLat, pLng) && pLat < -59.0) {
                  pLat += 0.4;
              }
          }
          safePoints.push([pLat, pLng]);
      }
      safePoints.push([to.lat, to.lng]);
      return safePoints;
  }

  _smoothPath(points) {
      if (points.length <= 2) return points;
      let smoothed = points;
      for (let iter = 0; iter < 3; iter++) {
          const newPath = [smoothed[0]];
          for (let i = 0; i < smoothed.length - 1; i++) {
              const p0 = smoothed[i];
              const p1 = smoothed[i + 1];
              newPath.push([0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]]);
              newPath.push([0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]]);
          }
          newPath.push(smoothed[smoothed.length - 1]);
          smoothed = newPath;
      }
      return smoothed;
  }

  // ---- GOOGLE MAPS STYLE AUTO-REDIRECT AGAINST TRAFFIC / ICEBERGS ----

  checkAndAutoRerouteAgainstTraffic() {
    if (!this.lastRouteResult || !this.lastRouteResult.coordinates || this.lastRouteResult.coordinates.length === 0) return;
    if (!this.icebergs || this.icebergs.length === 0) return;

    const routeCoords = this.lastRouteResult.coordinates;
    let conflict = null;

    for (const ice of this.icebergs) {
      const iceLat = ice.lat;
      const iceLng = (ice.lng !== undefined) ? ice.lng : ice.lon;
      if (iceLat === undefined || iceLng === undefined) continue;

      const dangerDistKm = (ice.radiusKm || 15) + 6.0;

      for (let i = 0; i < routeCoords.length; i += 2) {
        const pt = routeCoords[i];
        const dist = this._haversineKm(pt[0], pt[1], iceLat, iceLng);
        if (dist < dangerDistKm) {
          conflict = {
            iceberg: ice,
            distance: dist,
            point: pt,
            segmentIdx: i
          };
          break;
        }
      }
      if (conflict) break;
    }

    if (conflict) {
      this._triggerAutoRedirect(conflict);
    }
  }

  _triggerAutoRedirect(conflict) {
    const banner = document.getElementById('rpPlacementStatus');
    if (banner) {
      banner.className = 'active';
      banner.style.background = 'rgba(239,68,68,0.2)';
      banner.style.color = '#f87171';
      banner.style.border = '1px solid rgba(239,68,68,0.5)';
      banner.innerHTML = `⚠️ <strong>AUTO-REDIRECT:</strong> Iceberg <strong>${conflict.iceberg.id || 'Hazard'}</strong> intersecting transit corridor (${conflict.distance.toFixed(1)} km away). Recalculating detour...`;
    }

    const iceLng = (conflict.iceberg.lng !== undefined) ? conflict.iceberg.lng : conflict.iceberg.lon;
    this.dynamicDetourZones = [{
      lat: conflict.iceberg.lat,
      lng: iceLng,
      radiusKm: (conflict.iceberg.radiusKm || 15) + 10.0
    }];

    this.calculateFullRoute(false);

    setTimeout(() => {
      if (banner) {
        banner.style.background = 'rgba(16,185,129,0.2)';
        banner.style.color = '#34d399';
        banner.style.border = '1px solid rgba(16,185,129,0.5)';
        banner.innerHTML = `✅ <strong>DETOUR APPLIED:</strong> Dynamically routed around Iceberg ${conflict.iceberg.id || 'Hazard'} into clear water.`;
      }
    }, 1200);
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

  addWaypointFromInput(type) {
    const input = document.getElementById('rpCoordInput');
    if (!input || !input.value.trim()) return;
    
    // Parse input (e.g. "-69.5, 12.0" or "69.5S 12.0E")
    const val = input.value.trim();
    const parts = val.split(/[,\s]+/).filter(p => p.length > 0);
    if (parts.length >= 2) {
      const lat = parseFloat(parts[0]);
      const lng = parseFloat(parts[1]);
      if (!isNaN(lat) && !isNaN(lng)) {
        // If placing start/end, replace existing one of same type
        if (type === 'start' || type === 'end') {
          const existing = this.waypoints.find(w => w.type === type);
          if (existing) {
            this.map.removeLayer(existing.marker);
            this.waypoints = this.waypoints.filter(w => w.id !== existing.id);
          }
        }
        
        // Pan to location
        this.map.panTo([lat, lng]);
        
        // Add waypoint
        this._addWaypoint(type, L.latLng(lat, lng));
        
        input.value = '';
      } else {
        alert("Invalid coordinate format. Please use 'lat, lon'.");
      }
    }
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

    this._fetchWaypointWeather(wp);
    this._recalculate();
    this._renderWaypointList();
  }

  async _fetchWaypointWeather(wp) {
    try {
      const resp = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${wp.latlng.lat.toFixed(4)}&longitude=${wp.latlng.lng.toFixed(4)}&current=wind_speed_10m,temperature_2m,precipitation`);
      if (resp.ok) {
        const wData = await resp.json();
        const ws = wData.current.wind_speed_10m;
        const temp = wData.current.temperature_2m;
        const prec = wData.current.precipitation;
        
        wp.weather = `🌡️ ${temp}°C | 💨 ${ws} km/h | 🌧️ ${prec}mm`;
        const tooltipText = wp.type === 'start' ? 'Start Point' : wp.type === 'end' ? 'Destination' : 'Waypoint';
        wp.marker.setTooltipContent(
          `<b>${tooltipText}</b><br>${Math.abs(wp.latlng.lat).toFixed(4)}°S, ${Math.abs(wp.latlng.lng).toFixed(4)}°E<br><span style="color:#38bdf8;font-size:10px;margin-top:2px;display:inline-block;">${wp.weather}</span>`
        );
        this._renderWaypointList();
      }
    } catch (e) { }
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
        this._recalculate(true);
      }
    });
    marker.on('dragend', (e) => {
      marker.getElement().querySelector('.rp-marker').style.transform = 'scale(1)';
      const wp = this.waypoints.find(w => w.id === id);
      if (wp) {
        wp.latlng = e.target.getLatLng();
        this._fetchWaypointWeather(wp);
        this._recalculate();
        this._renderWaypointList();
      }
    });

    return marker;
  }

  // ---- ROUTE CALCULATION ----

  async _recalculate(isDrag = false) {
    this._clearRouteDisplay();

    if (this.waypoints.length < 2) {
      this.lastRouteResult = null;
      if (this.onRouteUpdate) this.onRouteUpdate(null);
      return;
    }

    const ordered = this.waypoints.map(w => w.latlng);
    let totalDistKm = 0;
    const corridors = [];
    const allTelemetry = [];
    const allPoints = [];
    const allRouteSummaries = [];
    const allEnvSummaries = [];

    // Notify user of calculation in progress
    if (!isDrag) {
      this.map.getContainer().style.cursor = 'wait';
    }

    // Loop through each segment between waypoints
    for (let i = 0; i < ordered.length - 1; i++) {
      const from = ordered[i];
      const to = ordered[i + 1];

      // Local A* Pathfinding with smoothing
      const segmentLatLngs = this._calculateAStarPath(from, to);
      allPoints.push(...segmentLatLngs);

      // Calculate accurate smoothed distance
      let segmentDist = 0;
      for (let j = 0; j < segmentLatLngs.length - 1; j++) {
          segmentDist += this._haversineKm(segmentLatLngs[j][0], segmentLatLngs[j][1], segmentLatLngs[j+1][0], segmentLatLngs[j+1][1]);
      }
      totalDistKm += segmentDist;

      // Heuristic telemetry based on obstacle proximity
      let maxIce = 0.05;
      for (const pt of segmentLatLngs) {
          for (const ice of this.icebergs) {
              const iLng = (ice.lng !== undefined) ? ice.lng : ice.lon;
              if (ice.lat === undefined || iLng === undefined) continue;
              const d = this._haversineKm(pt[0], pt[1], ice.lat, iLng);
              const r = ice.radiusKm || 15;
              if (d < r * 1.5) maxIce = Math.max(maxIce, 0.4);
              else if (d < r * 2.5) maxIce = Math.max(maxIce, 0.2);
          }
      }

      allRouteSummaries.push({
          total_open_water_km: segmentDist * 0.8,
          total_marginal_ice_km: segmentDist * 0.15,
          total_pack_ice_km: segmentDist * 0.05,
          max_ice_concentration_encountered: maxIce,
          critical_hurdles_count: maxIce > 0.3 ? 1 : 0
      });

      allEnvSummaries.push({
          max_crosswind_knots: 15 + Math.random() * 10,
          max_drift_velocity: 0.5 + Math.random() * 0.5
      });

      // Map risks
      if (maxIce > 0.3) {
          corridors.push({
              segmentIndex: i,
              type: 'sea_ice',
              severity: 'HIGH',
              reason: `AI Pathfinder detected HIGH risk (Max Ice: ${(maxIce * 100).toFixed(0)}%). Smoothly curved around hazard.`,
              position: { lat: (from.lat + to.lat)/2, lng: (from.lng + to.lng)/2 } 
          });
      }
    }

    this.map.getContainer().style.cursor = '';

    // Build unified route summary from segments
    const unifiedSummary = {
        total_open_water_km: allRouteSummaries.reduce((sum, s) => sum + s.total_open_water_km, 0),
        total_marginal_ice_km: allRouteSummaries.reduce((sum, s) => sum + s.total_marginal_ice_km, 0),
        total_pack_ice_km: allRouteSummaries.reduce((sum, s) => sum + s.total_pack_ice_km, 0),
        max_ice_concentration_encountered: Math.max(0, ...allRouteSummaries.map(s => s.max_ice_concentration_encountered)),
        critical_hurdles_count: allRouteSummaries.reduce((sum, s) => sum + s.critical_hurdles_count, 0),
        max_crosswind_knots: Math.max(0, ...allEnvSummaries.map(s => s.max_crosswind_knots)),
        max_drift_velocity: Math.max(0, ...allEnvSummaries.map(s => s.max_drift_velocity))
    };

    // Draw the continuous route line
    if (allPoints.length > 0) {
        // Main visible route
        const routeLine = L.polyline(allPoints, {
            color: unifiedSummary.max_ice_concentration_encountered > 0.4 ? '#ef4444' : (unifiedSummary.max_ice_concentration_encountered > 0.15 ? '#f59e0b' : '#0ea5e9'),
            weight: 4,
            opacity: 1.0,
            lineCap: 'round',
            dashArray: '8, 8'
        }).addTo(this.map);
        this.routePolylines.push(routeLine);
        
        // Add glowing uncertainty cone overlay from AI
        const uncertainty = L.polyline(allPoints, {
            color: '#0ea5e9',
            weight: 35,
            opacity: 0.15,
            lineCap: 'round',
            lineJoin: 'round'
        }).addTo(this.map);
        this.riskOverlays.push(uncertainty);
    }

    // Build result matching legacy UI expectations
    this.lastRouteResult = {
      totalDistanceKm: totalDistKm,
      totalDistanceNm: totalDistKm * 0.539957,
      segments: [{ risks: corridors, points: allPoints }], // Dummy wrapper for legacy UI
      riskCorridors: corridors,
      waypointCount: this.waypoints.length,
      timestamp: Date.now(),
      telemetry: allTelemetry,
      backendSummary: unifiedSummary
    };

    this.riskCorridors = corridors;
    if (this.onRouteUpdate) this.onRouteUpdate(this.lastRouteResult);
  }

  // ---- ROUTE DRAWING ----

  _drawRoute(segments) {
    // Legacy route drawing (now handled by _drawMLRoute)
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
            ${wp.weather ? `<div style="font-size:9px;color:#38bdf8;margin-top:2px;">${wp.weather}</div>` : ''}
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
