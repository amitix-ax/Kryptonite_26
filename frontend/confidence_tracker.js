/* Confidence & Auto-Refresh Tracker — Antarctic DSS Data Staleness Monitor */

class ConfidenceTracker {
  constructor(onRefresh) {
    this.onRefresh = onRefresh; // callback triggered on each auto-refresh cycle

    // Configuration
    this.refreshIntervalMs = 15 * 60 * 1000; // 15 minutes default
    this.maxConfidence = 98;                   // max confidence at fresh data
    this.decayHalfLifeMs = 30 * 60 * 1000;     // confidence halves in 30 min
    this.criticalThreshold = 30;
    this.warningThreshold = 60;

    // State
    this.lastUpdateTimestamp = Date.now();
    this.currentConfidence = this.maxConfidence;
    this.refreshTimer = null;
    this.decayTimer = null;
    this.nextRefreshAt = null;
    this.refreshCount = 0;
    this.isRefreshing = false;

    this._startDecayTick();
    this._startRefreshCycle();
    this._bindControls();
  }

  // ---- PUBLIC API ----

  markDataUpdated() {
    this.lastUpdateTimestamp = Date.now();
    this.currentConfidence = this.maxConfidence;
    this.refreshCount++;
    this.isRefreshing = false;
    this._updateDisplay();
  }

  setRefreshInterval(minutes) {
    this.refreshIntervalMs = minutes * 60 * 1000;
    this._startRefreshCycle();
    // Update dropdown if exists
    const sel = document.getElementById('ctRefreshInterval');
    if (sel) sel.value = minutes.toString();
  }

  getConfidence() {
    return this.currentConfidence;
  }

  getTimeSinceUpdate() {
    return Date.now() - this.lastUpdateTimestamp;
  }

  destroy() {
    if (this.decayTimer) clearInterval(this.decayTimer);
    if (this.refreshTimer) clearInterval(this.refreshTimer);
  }

  // ---- DECAY LOGIC ----

  _startDecayTick() {
    if (this.decayTimer) clearInterval(this.decayTimer);

    this.decayTimer = setInterval(() => {
      const elapsed = Date.now() - this.lastUpdateTimestamp;

      // Exponential decay: C(t) = C_max * (0.5)^(t / halfLife)
      this.currentConfidence = Math.max(
        5,
        this.maxConfidence * Math.pow(0.5, elapsed / this.decayHalfLifeMs)
      );

      this._updateDisplay();
    }, 2000); // tick every 2 seconds
  }

  // ---- AUTO-REFRESH ----

  _startRefreshCycle() {
    if (this.refreshTimer) clearInterval(this.refreshTimer);

    this.nextRefreshAt = Date.now() + this.refreshIntervalMs;

    this.refreshTimer = setInterval(() => {
      this.isRefreshing = true;
      this._updateDisplay();

      // Trigger the refresh callback
      if (this.onRefresh) {
        this.onRefresh();
      }

      // After a short delay, mark as updated (simulating data fetch)
      setTimeout(() => {
        this.markDataUpdated();
        this.nextRefreshAt = Date.now() + this.refreshIntervalMs;
      }, 1500);
    }, this.refreshIntervalMs);
  }

  // ---- UI BINDING ----

  _bindControls() {
    const sel = document.getElementById('ctRefreshInterval');
    if (sel) {
      sel.addEventListener('change', () => {
        this.setRefreshInterval(parseInt(sel.value, 10));
      });
    }

    const manualBtn = document.getElementById('ctManualRefresh');
    if (manualBtn) {
      manualBtn.addEventListener('click', () => {
        this.isRefreshing = true;
        this._updateDisplay();
        if (this.onRefresh) this.onRefresh();
        setTimeout(() => {
          this.markDataUpdated();
          this.nextRefreshAt = Date.now() + this.refreshIntervalMs;
        }, 1200);
      });
    }
  }

  // ---- DISPLAY ----

  _updateDisplay() {
    this._updateBadge();
    this._updateConfidenceBar();
    this._updateCountdown();
    this._updateLastUpdateText();
  }

  _updateBadge() {
    const badge = document.getElementById('ctBadge');
    if (!badge) return;

    const conf = this.currentConfidence;

    if (this.isRefreshing) {
      badge.className = 'ct-badge ct-refreshing';
      badge.innerHTML = '<span class="ct-spinner"></span> REFRESHING…';
    } else if (conf >= this.warningThreshold) {
      badge.className = 'ct-badge ct-fresh';
      badge.innerHTML = `<span class="ct-dot ct-dot-fresh"></span> FRESH ${Math.round(conf)}%`;
    } else if (conf >= this.criticalThreshold) {
      badge.className = 'ct-badge ct-warning';
      badge.innerHTML = `<span class="ct-dot ct-dot-warning"></span> AGING ${Math.round(conf)}%`;
    } else {
      badge.className = 'ct-badge ct-stale';
      badge.innerHTML = `<span class="ct-dot ct-dot-stale"></span> STALE ${Math.round(conf)}%`;
    }
  }

  _updateConfidenceBar() {
    const bar = document.getElementById('ctConfBar');
    const pct = document.getElementById('ctConfPct');
    if (!bar) return;

    const conf = this.currentConfidence;
    bar.style.width = conf + '%';

    if (conf >= this.warningThreshold) {
      bar.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
      bar.style.boxShadow = '0 0 10px rgba(16,185,129,0.4)';
    } else if (conf >= this.criticalThreshold) {
      bar.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
      bar.style.boxShadow = '0 0 10px rgba(245,158,11,0.4)';
    } else {
      bar.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
      bar.style.boxShadow = '0 0 10px rgba(239,68,68,0.4)';
    }

    if (pct) pct.textContent = Math.round(conf) + '%';
  }

  _updateCountdown() {
    const el = document.getElementById('ctCountdown');
    if (!el || !this.nextRefreshAt) return;

    const remaining = Math.max(0, this.nextRefreshAt - Date.now());
    const min = Math.floor(remaining / 60000);
    const sec = Math.floor((remaining % 60000) / 1000);
    el.textContent = `Next refresh: ${min}:${sec.toString().padStart(2, '0')}`;
  }

  _updateLastUpdateText() {
    const el = document.getElementById('ctLastUpdate');
    if (!el) return;

    const elapsed = Date.now() - this.lastUpdateTimestamp;
    const sec = Math.floor(elapsed / 1000);
    const min = Math.floor(sec / 60);

    if (min < 1) {
      el.textContent = `Updated ${sec}s ago`;
    } else if (min < 60) {
      el.textContent = `Updated ${min}m ${sec % 60}s ago`;
    } else {
      const hrs = Math.floor(min / 60);
      el.textContent = `Updated ${hrs}h ${min % 60}m ago`;
    }

    // Color coding
    if (this.currentConfidence >= this.warningThreshold) {
      el.style.color = '#10b981';
    } else if (this.currentConfidence >= this.criticalThreshold) {
      el.style.color = '#f59e0b';
    } else {
      el.style.color = '#ef4444';
    }
  }
}
