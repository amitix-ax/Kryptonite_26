/* Fuel & Vessel Calculator — Antarctic DSS Route Fuel Estimation */

class FuelCalculator {
  constructor(onUpdate) {
    this.onUpdate = onUpdate; // callback(fuelResult)

    // Default vessel: S.A. Agulhas II (typical Antarctic supply vessel)
    this.vesselSpecs = {
      name: 'S.A. Agulhas II',
      type: 'icebreaker',
      displacement: 13687,      // tonnes
      enginePower: 12000,        // kW (2×6000)
      cruisingSpeed: 14,         // knots
      fuelRate: 2.8,             // tonnes/hr at cruising speed
      iceClass: 'PC5',           // Polar Class
      fuelPrice: 850             // $/tonne (marine diesel)
    };

    // Ice class speed reduction factors
    this.iceClassFactors = {
      'PC1': 0.85, 'PC2': 0.80, 'PC3': 0.75, 'PC4': 0.70,
      'PC5': 0.65, 'PC6': 0.55, 'PC7': 0.45,
      'IA Super': 0.60, 'IA': 0.50, 'IB': 0.40, 'IC': 0.30,
      'None': 0.15
    };

    // Extra fuel burn multiplier in ice
    this.iceFuelMultiplier = {
      'PC1': 1.15, 'PC2': 1.20, 'PC3': 1.30, 'PC4': 1.40,
      'PC5': 1.50, 'PC6': 1.70, 'PC7': 2.00,
      'IA Super': 1.60, 'IA': 1.80, 'IB': 2.10, 'IC': 2.50,
      'None': 3.00
    };

    this.lastResult = null;
    this._bindForm();
  }

  // ---- PUBLIC API ----

  updateSpecs(specs) {
    Object.assign(this.vesselSpecs, specs);
    this._syncFormToSpecs();
  }

  calculateForRoute(routeData) {
    if (!routeData) {
      this.lastResult = null;
      this._renderResult(null);
      return null;
    }

    const specs = this.vesselSpecs;
    const totalNm = routeData.totalDistanceNm;

    // Separate ice and open-water segments
    let iceNm = 0;
    let openNm = 0;
    routeData.segments.forEach(seg => {
      const isIce = seg.risks.some(r => r.type === 'sea_ice');
      if (isIce) iceNm += seg.distanceNm;
      else openNm += seg.distanceNm;
    });

    // Speed calculations
    const openSpeedKts = specs.cruisingSpeed;
    const iceSpeedFactor = this.iceClassFactors[specs.iceClass] || 0.50;
    const iceSpeedKts = Math.max(openSpeedKts * iceSpeedFactor, 2); // min 2 knots in ice

    // Time calculations
    const openTimeHrs = openNm / openSpeedKts;
    const iceTimeHrs = iceNm > 0 ? iceNm / iceSpeedKts : 0;
    const totalTimeHrs = openTimeHrs + iceTimeHrs;

    // Fuel calculations
    const iceFuelMult = this.iceFuelMultiplier[specs.iceClass] || 1.5;
    const openFuel = openTimeHrs * specs.fuelRate;
    const iceFuel = iceTimeHrs * specs.fuelRate * iceFuelMult;
    const totalFuel = openFuel + iceFuel;

    // Cost
    const totalCost = totalFuel * specs.fuelPrice;

    // Risk count
    const totalRisks = routeData.riskCorridors.length;
    const criticalRisks = routeData.riskCorridors.filter(r => r.severity === 'CRITICAL').length;
    const highRisks = routeData.riskCorridors.filter(r => r.severity === 'HIGH').length;

    this.lastResult = {
      totalDistanceNm: routeData.totalDistanceNm,
      totalDistanceKm: routeData.totalDistanceKm,
      openWaterNm: openNm,
      iceZoneNm: iceNm,
      openWaterSpeed: openSpeedKts,
      iceZoneSpeed: iceSpeedKts,
      openWaterTimeHrs: openTimeHrs,
      iceZoneTimeHrs: iceTimeHrs,
      totalTimeHrs,
      totalTimeDays: totalTimeHrs / 24,
      openWaterFuel: openFuel,
      iceZoneFuel: iceFuel,
      totalFuelTonnes: totalFuel,
      totalCostUSD: totalCost,
      fuelPricePerTonne: specs.fuelPrice,
      iceClass: specs.iceClass,
      vesselName: specs.name,
      totalRisks,
      criticalRisks,
      highRisks,
      timestamp: Date.now()
    };

    this._renderResult(this.lastResult);
    if (this.onUpdate) this.onUpdate(this.lastResult);
    return this.lastResult;
  }

  // ---- FORM BINDING ----

  _bindForm() {
    const fields = ['fcVesselName', 'fcVesselType', 'fcDisplacement', 'fcEnginePower',
                     'fcCruisingSpeed', 'fcFuelRate', 'fcIceClass', 'fcFuelPrice'];

    fields.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('input', () => this._readForm());
        el.addEventListener('change', () => this._readForm());
      }
    });

    // Initialize form with defaults
    this._syncFormToSpecs();
  }

  _readForm() {
    const get = (id) => {
      const el = document.getElementById(id);
      return el ? el.value : null;
    };
    const getNum = (id) => {
      const v = parseFloat(get(id));
      return isNaN(v) ? null : v;
    };

    this.vesselSpecs.name = get('fcVesselName') || this.vesselSpecs.name;
    this.vesselSpecs.type = get('fcVesselType') || this.vesselSpecs.type;
    this.vesselSpecs.displacement = getNum('fcDisplacement') || this.vesselSpecs.displacement;
    this.vesselSpecs.enginePower = getNum('fcEnginePower') || this.vesselSpecs.enginePower;
    this.vesselSpecs.cruisingSpeed = getNum('fcCruisingSpeed') || this.vesselSpecs.cruisingSpeed;
    this.vesselSpecs.fuelRate = getNum('fcFuelRate') || this.vesselSpecs.fuelRate;
    this.vesselSpecs.iceClass = get('fcIceClass') || this.vesselSpecs.iceClass;
    this.vesselSpecs.fuelPrice = getNum('fcFuelPrice') || this.vesselSpecs.fuelPrice;

    // Re-fire calculation if we have route data
    if (this.onUpdate) this.onUpdate(null); // signal specs changed
  }

  _syncFormToSpecs() {
    const set = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.value = val;
    };
    set('fcVesselName', this.vesselSpecs.name);
    set('fcVesselType', this.vesselSpecs.type);
    set('fcDisplacement', this.vesselSpecs.displacement);
    set('fcEnginePower', this.vesselSpecs.enginePower);
    set('fcCruisingSpeed', this.vesselSpecs.cruisingSpeed);
    set('fcFuelRate', this.vesselSpecs.fuelRate);
    set('fcIceClass', this.vesselSpecs.iceClass);
    set('fcFuelPrice', this.vesselSpecs.fuelPrice);
  }

  // ---- RESULT RENDERING ----

  _renderResult(result) {
    const container = document.getElementById('fcResultPanel');
    if (!container) return;

    if (!result) {
      container.innerHTML = `
        <div class="fc-empty">
          <div class="fc-empty-icon">⛽</div>
          <div class="fc-empty-text">Place route markers to calculate fuel & transit estimates</div>
        </div>`;
      return;
    }

    const riskBadge = result.criticalRisks > 0
      ? `<span class="fc-badge fc-badge-critical">⚠ ${result.criticalRisks} CRITICAL</span>`
      : result.highRisks > 0
      ? `<span class="fc-badge fc-badge-high">⚠ ${result.highRisks} HIGH RISK</span>`
      : `<span class="fc-badge fc-badge-ok">✓ LOW RISK</span>`;

    container.innerHTML = `
      <div class="fc-summary-header">
        <div class="fc-vessel-tag">🚢 ${result.vesselName}</div>
        ${riskBadge}
      </div>
      <div class="fc-metric-grid">
        <div class="fc-metric">
          <div class="fc-metric-label">TOTAL DISTANCE</div>
          <div class="fc-metric-value">${result.totalDistanceNm.toFixed(1)} <small>nm</small></div>
          <div class="fc-metric-sub">${result.totalDistanceKm.toFixed(1)} km</div>
        </div>
        <div class="fc-metric">
          <div class="fc-metric-label">TRANSIT TIME</div>
          <div class="fc-metric-value">${result.totalTimeHrs.toFixed(1)} <small>hrs</small></div>
          <div class="fc-metric-sub">${result.totalTimeDays.toFixed(2)} days</div>
        </div>
        <div class="fc-metric fc-metric-fuel">
          <div class="fc-metric-label">FUEL REQUIRED</div>
          <div class="fc-metric-value">${result.totalFuelTonnes.toFixed(1)} <small>t</small></div>
          <div class="fc-metric-sub">$${(result.totalCostUSD / 1000).toFixed(1)}K USD</div>
        </div>
        <div class="fc-metric">
          <div class="fc-metric-label">ICE CLASS</div>
          <div class="fc-metric-value">${result.iceClass}</div>
          <div class="fc-metric-sub">Ice speed: ${result.iceZoneSpeed.toFixed(1)} kts</div>
        </div>
      </div>
      ${result.iceZoneNm > 0 ? `
      <div class="fc-breakdown">
        <div class="fc-breakdown-row">
          <span class="fc-bd-label">Open water</span>
          <span class="fc-bd-val">${result.openWaterNm.toFixed(1)} nm · ${result.openWaterTimeHrs.toFixed(1)} h · ${result.openWaterFuel.toFixed(1)} t fuel</span>
        </div>
        <div class="fc-breakdown-row fc-breakdown-ice">
          <span class="fc-bd-label">❄️ Ice zone</span>
          <span class="fc-bd-val">${result.iceZoneNm.toFixed(1)} nm · ${result.iceZoneTimeHrs.toFixed(1)} h · ${result.iceZoneFuel.toFixed(1)} t fuel</span>
        </div>
      </div>` : ''}
    `;
  }
}
