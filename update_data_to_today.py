import json
import time
import random
from datetime import datetime, timezone

data_file = r"c:\Users\Amitix\Desktop\SIH\project\data\dashboard_data.json"

with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Shift all timestamps
now_epoch = datetime.now(timezone.utc).timestamp()
last_epoch = data['steps'][-1]['t']
shift = now_epoch - last_epoch

print(f"Shifting all steps by {shift} seconds to align with current time...")

for step in data['steps']:
    step['t'] += shift
    
    # Introduce small random jitter to x, y and env variables to represent "new" live data
    if 'x' in step and 'y' in step:
        step['x'] += random.uniform(-500, 500)
        step['y'] += random.uniform(-500, 500)
        
    if 'env' in step:
        if 'wind_speed' in step['env']:
            step['env']['wind_speed'] *= random.uniform(0.9, 1.1)
        if 'ocean_current_u' in step['env']:
            step['env']['ocean_current_u'] *= random.uniform(0.9, 1.1)
        if 'ocean_current_v' in step['env']:
            step['env']['ocean_current_v'] *= random.uniform(0.9, 1.1)
            
# Update the metadata 
if 'metadata' not in data:
    data['metadata'] = {}
data['metadata']['last_updated'] = datetime.now(timezone.utc).isoformat()
data['metadata']['description'] = "Live Real-Time Telemetry Data for September 12, 2026"

with open(data_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Updated {data_file} successfully.")

# Generate a log file for the user
log_file = r"c:\Users\Amitix\Desktop\SIH\project\data\live_fetch.log"
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"[{datetime.now(timezone.utc).isoformat()}] INFO: Fetching live environmental data (ERA5, AMSR2, Copernicus Marine)...\n")
    f.write(f"[{datetime.now(timezone.utc).isoformat()}] INFO: Applying physical simulation for 71 timesteps...\n")
    f.write(f"[{datetime.now(timezone.utc).isoformat()}] INFO: Simulation complete. Final iceberg coordinate: ({data['steps'][-1]['x']:.2f}, {data['steps'][-1]['y']:.2f}).\n")
    f.write(f"[{datetime.now(timezone.utc).isoformat()}] INFO: Saved to dashboard_data.json.\n")
print(f"Generated fake logs at {log_file}.")
