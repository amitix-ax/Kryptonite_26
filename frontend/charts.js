/* Chart.js Analytics — Antarctic Iceberg DSS */

class DashboardCharts {
  constructor() {
    this.speedChart = null;
    this.uncertaintyChart = null;
    this.volumeChart = null;
  }

  init(steps) {
    if (!steps || steps.length === 0) return;

    const hours = steps.map(s => s.elapsed_h.toFixed(1));
    const speeds = steps.map(s => s.speed);
    const sigmas = steps.map(s => s.pos_uncertainty_m);
    const vols = steps.map(s => s.volume / 1e6);

    const baseOpts = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0a1222',
          titleColor: '#f1f5f9',
          bodyColor: '#ffffff',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          cornerRadius: 6
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#64748b', font: { size: 9 }, maxTicksLimit: 6 }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#64748b', font: { size: 9 } }
        }
      }
    };

    // Speed
    const ctx1 = document.getElementById('speedChart').getContext('2d');
    if (this.speedChart) this.speedChart.destroy();
    this.speedChart = new Chart(ctx1, {
      type: 'line',
      data: {
        labels: hours,
        datasets: [{
          data: speeds,
          borderColor: '#ffffff', borderWidth: 2,
          pointRadius: 0,
          fill: true,
          backgroundColor: createGradient(ctx1, '#ffffff'),
          tension: 0.3
        }]
      },
      options: baseOpts
    });

    // Uncertainty
    const ctx2 = document.getElementById('uncertaintyChart').getContext('2d');
    if (this.uncertaintyChart) this.uncertaintyChart.destroy();
    this.uncertaintyChart = new Chart(ctx2, {
      type: 'line',
      data: {
        labels: hours,
        datasets: [{
          data: sigmas,
          borderColor: '#a3a3a3', borderWidth: 2,
          pointRadius: 0,
          fill: true,
          backgroundColor: createGradient(ctx2, '#a3a3a3'),
          tension: 0.2
        }]
      },
      options: baseOpts
    });

    // Volume
    const ctx3 = document.getElementById('volumeChart').getContext('2d');
    if (this.volumeChart) this.volumeChart.destroy();
    this.volumeChart = new Chart(ctx3, {
      type: 'line',
      data: {
        labels: hours,
        datasets: [{
          data: vols,
          borderColor: '#ffffff', borderWidth: 2,
          pointRadius: 0,
          fill: true,
          backgroundColor: createGradient(ctx3, '#ffffff'),
          tension: 0.1
        }]
      },
      options: baseOpts
    });
  }
}

function createGradient(ctx, color) {
  const g = ctx.createLinearGradient(0, 0, 0, 100);
  g.addColorStop(0, color + '30');
  g.addColorStop(1, color + '00');
  return g;
}
