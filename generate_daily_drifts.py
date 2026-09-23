import json
import random
import os
from datetime import datetime, timedelta, timezone

data_file = r"c:\Users\Amitix\Desktop\SIH\project\data\dashboard_data.json"
out_dir = r"c:\Users\Amitix\Desktop\SIH\project\data"

def generate_history():
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    
    last_epoch = data['steps'][-1]['t']
    
    for i in range(365):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        target_file = os.path.join(out_dir, f"drift_history_{date_str}.json")
        
        target_epoch = current_date.timestamp()
        time_shift = target_epoch - last_epoch
        
        day_of_year = current_date.timetuple().tm_yday
        x_shift = (day_of_year - 250) * 1500
        y_shift = (day_of_year - 250) * 500
        
        day_data = json.loads(json.dumps(data))
        
        for step in day_data['steps']:
            step['t'] += time_shift
            
            if 'x' in step and 'y' in step:
                step['x'] += x_shift + random.uniform(-500, 500)
                step['y'] += y_shift + random.uniform(-500, 500)
                step['lat'] += y_shift / 111000.0
                step['lon'] += x_shift / (111000.0 * 0.4) 
                
        if 'metadata' not in day_data:
            day_data['metadata'] = {}
        day_data['metadata']['description'] = f"Historical Drift: Iceberg Simulation ({date_str})"
        day_data['metadata']['last_updated'] = current_date.isoformat()
        
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(day_data, f, separators=(',', ':'))

if __name__ == "__main__":
    generate_history()
    print("Generated 365 daily files.")
