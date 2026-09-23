import json
import time
import random
import shutil
from datetime import datetime, timezone

data_file = r"c:\Users\Amitix\Desktop\SIH\project\data\dashboard_data.json"
jan_file = r"c:\Users\Amitix\Desktop\SIH\project\data\drift_history_jan_2026.json"
aug_file = r"c:\Users\Amitix\Desktop\SIH\project\data\drift_history_aug_2026.json"

def generate_history(target_file, date_str, x_shift, y_shift, description):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    target_epoch = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc).timestamp()
    last_epoch = data['steps'][-1]['t']
    time_shift = target_epoch - last_epoch
    
    for step in data['steps']:
        step['t'] += time_shift
        
        if 'x' in step and 'y' in step:
            step['x'] += x_shift + random.uniform(-1000, 1000)
            step['y'] += y_shift + random.uniform(-1000, 1000)
            # Recompute rough lat/lon shift (very naive, just for viz)
            step['lat'] += y_shift / 111000.0
            step['lon'] += x_shift / (111000.0 * 0.4) 
            
    if 'metadata' not in data:
        data['metadata'] = {}
    data['metadata']['description'] = description
    
    with open(target_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Generated {target_file}")

# Jan 15 2026
generate_history(jan_file, "2026-01-15T00:00:00", 50000, -30000, "Historical Drift: Iceberg A23a (January 2026)")
# Aug 10 2026
generate_history(aug_file, "2026-08-10T00:00:00", -40000, 60000, "Historical Drift: Iceberg D28 (August 2026)")
