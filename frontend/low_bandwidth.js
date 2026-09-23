document.addEventListener('DOMContentLoaded', () => {
  const calcBtn = document.getElementById('lbCalcBtn');
  const outDiv = document.getElementById('lbOutput');
  const iceDiv = document.getElementById('lbIcebergs');

  // Simple haversine for text mode
  function calcDist(lat1, lon1, lat2, lon2) {
    const R = 3440.065; // nm
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  }

  calcBtn.addEventListener('click', () => {
    outDiv.innerText = "Calculating...";
    
    setTimeout(() => {
      try {
        const startRaw = document.getElementById('lbStart').value.split(',');
        const endRaw = document.getElementById('lbEnd').value.split(',');
        
        const sLat = parseFloat(startRaw[0]);
        const sLon = parseFloat(startRaw[1]);
        const eLat = parseFloat(endRaw[0]);
        const eLon = parseFloat(endRaw[1]);

        if (isNaN(sLat) || isNaN(eLat)) throw "Invalid coordinates";

        const distNm = calcDist(sLat, sLon, eLat, eLon);
        const speedKts = 12.0; // S.A. Agulhas avg open water speed
        const timeHrs = distNm / speedKts;
        const fuelTonnes = timeHrs * 4.2; // roughly 100t per 24h

        let report = `ROUTE SUMMARY\n`;
        report += `-----------------\n`;
        report += `Start : ${sLat.toFixed(4)}°, ${sLon.toFixed(4)}°\n`;
        report += `End   : ${eLat.toFixed(4)}°, ${eLon.toFixed(4)}°\n`;
        report += `Dist  : ${distNm.toFixed(1)} nautical miles\n`;
        report += `Est. Time: ${timeHrs.toFixed(1)} hrs\n`;
        report += `Est. Fuel: ${fuelTonnes.toFixed(1)} t (S.A. Agulhas II)\n\n`;
        
        if (sLat < -60 || eLat < -60) {
          report += `>> WARNING: Entering Southern Ocean (below 60°S).\n`;
          report += `>> Expect pack ice and severe wind conditions.\n`;
        }
        
        outDiv.innerText = report;

      } catch(e) {
        outDiv.innerText = "Error: " + e;
      }
    }, 300);
  });

  // Load a single static lightweight JSON request for icebergs
  fetch('../data/drift_history_2026-09-12.json')
    .then(r => r.json())
    .then(data => {
      if (!data.trajectories || data.trajectories.length === 0) {
         iceDiv.innerText = "No iceberg telemetry available.";
         return;
      }
      const t = data.trajectories;
      iceDiv.innerHTML = `<table>
        <tr><th>ID</th><th>LAT</th><th>LON</th><th>RISK</th></tr>
        ${t.slice(0, 5).map(ice => {
           const p = ice.path[0];
           const risk = ice.risk_level === 'High' ? '<span class="warn">HIGH</span>' : 'LOW';
           return `<tr><td>${ice.id.substring(0,6)}</td><td>${p.lat.toFixed(2)}</td><td>${p.lon.toFixed(2)}</td><td>${risk}</td></tr>`;
        }).join('')}
      </table>
      <div style="margin-top:10px; font-size:11px;">Displaying closest 5 threats from network.</div>`;
    })
    .catch(err => {
      iceDiv.innerText = "Network Error: Could not load iceberg data over tactical link.";
    });
});
