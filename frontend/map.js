/* Leaflet Map Controller — Antarctic Iceberg DSS */

class IcebergMap {
  constructor(elementId) {
    this.elementId = elementId;
    this.map = null;
    this.tileLayers = {};
    this.overlayLayers = {};
    this.trajectoryGhost = null;
    this.trajectoryActive = null;
    this.icebergMarker = null;
    this.uncertaintyCircle = null;
    this.stationMarkers = [];
    this.vesselMarkers = [];
    this.icebergDbMarkers = [];
    this.routePolyline = null;
    this.seaIcePolygon = null;
    this.riskZones = [];
  }

  init(centerLat = -69.5, centerLon = 30.0, zoom = 4) {
    this.map = L.map(this.elementId, {
      center: [centerLat, centerLon],
      zoom: zoom,
      minZoom: 2,
      maxBounds: [
        [-90, -180],
        [90, 180]
      ],
      maxBoundsViscosity: 1.0,
      zoomControl: false,
      attributionControl: false
    });

    L.control.zoom({ position: 'topright' }).addTo(this.map);
    L.control.attribution({ position: 'bottomright', prefix: false }).addTo(this.map);

    // ---- TILE LAYERS ----

    // Google Satellite
    this.tileLayers.satellite = L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
      attribution: 'Imagery &copy; Google', maxZoom: 18
    });

    // Google Terrain
    this.tileLayers.terrain = L.tileLayer('https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}', {
      attribution: 'Imagery &copy; Google', maxZoom: 18
    });

    // Google Hybrid
    this.tileLayers.hybrid = L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', {
      attribution: 'Imagery &copy; Google', maxZoom: 18
    });

    // MapTiler Satellite
    this.tileLayers.maptiler = L.tileLayer('https://api.maptiler.com/maps/satellite/{z}/{x}/{y}.jpg?key=D2OlNY0GTKXU8iVvUESX', {
      attribution: '&copy; MapTiler', maxZoom: 18
    });

    // Esri World Imagery
    this.tileLayers.esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri', maxZoom: 18
    });

    // Default: satellite
    this.tileLayers.satellite.addTo(this.map);

    // Layer switcher
    document.querySelectorAll('.layer-chip').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.layer-chip').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const key = btn.dataset.layer;
        Object.values(this.tileLayers).forEach(l => this.map.removeLayer(l));
        if (this.tileLayers[key]) this.tileLayers[key].addTo(this.map);
      });
    });
  }

  // ---- TRAJECTORY ----
  renderTrajectory(steps) {
    if (!steps || steps.length === 0) return;
    const pts = steps.map(s => [s.lat, s.lon]);

    if (this.trajectoryGhost) this.map.removeLayer(this.trajectoryGhost);
    if (this.trajectoryActive) this.map.removeLayer(this.trajectoryActive);

    // Ghost (predicted full path)
    this.trajectoryGhost = L.polyline(pts, {
      color: '#38bdf8', weight: 2, dashArray: '6,6', opacity: 0.4
    }).addTo(this.map);

    // Active (animated progress)
    this.trajectoryActive = L.polyline([], {
      color: '#38bdf8', weight: 4, opacity: 0.95
    }).addTo(this.map);

    // Fit bounds
    this.map.fitBounds(L.latLngBounds(pts), { padding: [60, 60], maxZoom: 8 });

    this.updateMarker(steps[0]);
  }

  // ---- SEA ICE ZONE ----
  renderSeaIce(bounds) {
    if (this.seaIcePolygon) this.map.removeLayer(this.seaIcePolygon);
    if (!bounds || bounds.length < 3) return;

    this.seaIcePolygon = L.polygon(bounds, {
      color: '#ffffff',
      weight: 1.5,
      dashArray: '4,6',
      fillColor: '#ffffff',
      fillOpacity: 0.10
    }).addTo(this.map);

    this.seaIcePolygon.bindTooltip(
      '<b>Antarctic Sea Ice Pack Zone</b><br>ERA5/AMSR2 derived boundary<br>SIC: 3-65% (variable)',
      { sticky: true }
    );
  }

  // ---- STATIONS, VESSELS, ROUTE ----
  renderStationsAndRoute(stations, route, vessels) {
    // Clear previous
    this.stationMarkers.forEach(m => this.map.removeLayer(m));
    this.stationMarkers = [];
    this.vesselMarkers.forEach(m => this.map.removeLayer(m));
    this.vesselMarkers = [];

    // Normalize stations
    let stList = [];
    if (Array.isArray(stations)) stList = stations;
    else if (stations && typeof stations === 'object') {
      stList = Object.entries(stations).map(([n, c]) => ({
        name: n, lon: Array.isArray(c) ? c[0] : c.lon, lat: Array.isArray(c) ? c[1] : c.lat
      }));
    }

    // Render stations
    stList.forEach(st => {
      const icon = L.divIcon({
        className: '',
        html: `<div style="
          background:rgba(255,255,255,0.15);
          color:#ffffff;
          font-weight:700; font-size:10px;
          padding:3px 8px; border-radius:6px;
          border:1.5px solid #ffffff;
          white-space:nowrap;
          box-shadow:0 0 14px rgba(255,255,255,0.3);
          backdrop-filter:blur(4px);
          display:flex; align-items:center; gap:4px;
        " class="dynamic-station"><div class="pulse-dot"></div>STN ${st.name}</div>`,
        iconSize: [140, 24], iconAnchor: [70, 12]
      });
      const m = L.marker([st.lat, st.lon], { icon }).addTo(this.map);
      m.bindPopup(`<b>Research Station</b><br><b>${st.name}</b><br>${st.lat.toFixed(4)}°S, ${st.lon.toFixed(4)}°E`);
      this.stationMarkers.push(m);
    });

    // Render vessels
    if (vessels && vessels.length) {
      vessels.forEach(v => {
        const icon = L.divIcon({
          className: '',
          html: `<div style="
            background:rgba(255,255,255,0.12);
            color:#ffffff;
            font-weight:600; font-size:9px;
            padding:2px 7px; border-radius:5px;
            border:1.5px solid #ffffff;
            white-space:nowrap;
            box-shadow:0 0 10px rgba(255,255,255,0.25);
            display:flex; align-items:center; gap:3px;
          " class="dynamic-vessel"><div class="pulse-dot"></div>VESSEL ${v.name}</div>`,
          iconSize: [160, 20], iconAnchor: [80, 10]
        });
        const m = L.marker([v.lat, v.lon], { icon }).addTo(this.map);
        m.bindPopup(`<b>${v.name}</b><br>Status: ${v.status}<br>Heading: ${v.heading}°`);
        this.vesselMarkers.push(m);
      });
    }

    // Route polyline
    if (this.routePolyline) this.map.removeLayer(this.routePolyline);
    let rc = [];
    if (Array.isArray(route) && route.length > 0) {
      if (typeof route[0] === 'object' && route[0].lat !== undefined) {
        rc = route.map(w => [w.lat, w.lon]);
      } else if (Array.isArray(route[0])) {
        rc = route.map(w => [w[1], w[0]]);
      }
    }
    if (rc.length > 1) {
      this.routePolyline = L.polyline(rc, {
        color: '#a3a3a3', weight: 3, dashArray: '6,10', opacity: 0.8
      }).addTo(this.map);
      this.routePolyline.bindTooltip('<b>Indian Antarctic Supply Corridor</b><br>Bharati ↔ Maitri', { sticky: true });
    }
  }

  // ---- NIC ICEBERG DATABASE MARKERS ----
  renderIcebergDatabase(icebergs) {
    this.icebergDbMarkers.forEach(m => this.map.removeLayer(m));
    this.icebergDbMarkers = [];

    if (!icebergs || !icebergs.length) return;

    icebergs.forEach(ib => {
      const sizeKm = Math.max(ib.length_km || 10, 5);
      const color = sizeKm > 50 ? '#ff0033' : sizeKm > 20 ? '#a3a3a3' : '#ffffff';

      // Iceberg shape icon
      const icon = L.divIcon({
        className: '',
        html: `<div style="position:relative;">
          <svg width="28" height="28" viewBox="0 0 28 28">
            <polygon points="14 2 26 22 2 22" fill="${color}" fill-opacity="0.6" stroke="${color}" stroke-width="1.5"/>
            <polygon points="14 8 22 22 6 22" fill="white" fill-opacity="0.25"/>
          </svg>
          <div style="position:absolute;top:-8px;left:28px;background:rgba(6,10,19,0.85);color:${color};font-size:9px;font-weight:700;padding:1px 5px;border-radius:3px;white-space:nowrap;border:1px solid ${color};font-family:'JetBrains Mono',monospace;">${ib.id}</div>
        </div>`,
        iconSize: [28, 28], iconAnchor: [14, 22]
      });

      const m = L.marker([ib.lat, ib.lon], { icon }).addTo(this.map);
      m.bindPopup(`<b>Iceberg ${ib.id}</b><br>Size: ${ib.length_km}×${ib.width_km || '?'} km<br>Position: ${ib.lat.toFixed(3)}°S, ${ib.lon.toFixed(3)}°E<br>Source: ${ib.source || 'NIC'}`);
      this.icebergDbMarkers.push(m);

      // Danger radius circle for large icebergs
      if (sizeKm > 30) {
        const dangerCircle = L.circle([ib.lat, ib.lon], {
          radius: sizeKm * 1000 * 2,
          color: '#ff0033', weight: 1, dashArray: '4,4',
          fillColor: '#ff0033', fillOpacity: 0.05
        }).addTo(this.map);
        dangerCircle.bindTooltip(`Danger Zone — Iceberg ${ib.id} (${sizeKm} km)`);
        this.icebergDbMarkers.push(dangerCircle);
      }
    });
  }

  // ---- STEP UPDATE ----
  updateStep(step, activeSteps) {
    if (!step) return;

    if (this.trajectoryActive) {
      this.trajectoryActive.setLatLngs(activeSteps.map(s => [s.lat, s.lon]));
    }
    this.updateMarker(step);
    this.updateUncertainty(step);
  }

  updateMarker(step) {
    const grounded = step.mode === 'GROUNDED';
    const color = grounded ? '#ff0033' : '#ffffff';
    const glow = grounded ? 'rgba(255,0,51,0.4)' : 'rgba(255,255,255,0.4)';

    const html = `
      <div style="transform:rotate(${step.heading}deg);transition:transform 0.3s ease;">
        <svg width="36" height="36" viewBox="0 0 32 32">
          <defs>
            <filter id="glow"><feGaussianBlur stdDeviation="2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          </defs>
          <polygon points="16 1 30 28 16 23 2 28" fill="${color}" fill-opacity="0.9" stroke="#fff" stroke-width="2" filter="url(#glow)"/>
        </svg>
      </div>`;

    const icon = L.divIcon({
      className: 'iceberg-marker',
      html: html,
      iconSize: [36, 36], iconAnchor: [18, 18]
    });

    if (!this.icebergMarker) {
      this.icebergMarker = L.marker([step.lat, step.lon], { icon, zIndexOffset: 1000 }).addTo(this.map);
    } else {
      this.icebergMarker.setLatLng([step.lat, step.lon]);
      this.icebergMarker.setIcon(icon);
    }
  }

  updateUncertainty(step) {
    const r = Math.max(step.pos_uncertainty_m || 0, 30);
    if (!this.uncertaintyCircle) {
      this.uncertaintyCircle = L.circle([step.lat, step.lon], {
        radius: r,
        color: '#ffffff', fillColor: '#ffffff',
        fillOpacity: 0.12, weight: 1.5, dashArray: '4,4'
      }).addTo(this.map);
    } else {
      this.uncertaintyCircle.setLatLng([step.lat, step.lon]);
      this.uncertaintyCircle.setRadius(r);
    }
  }
}
