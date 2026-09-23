import math
import uuid
import numpy as np

class Obstacle:
    def __init__(self, x, y, rcs, freeboard_height):
        self.x = x
        self.y = y
        self.rcs = rcs # Radar Cross Section
        self.freeboard_height = freeboard_height
        
        # Isostatic equilibrium calculation
        # rho_w * z_d = rho_i * H_ice
        # H_ice = h_f + z_d
        # rho_w * z_d = rho_i * (h_f + z_d)
        # z_d * (rho_w - rho_i) = rho_i * h_f
        # z_d = (rho_i * h_f) / (rho_w - rho_i)
        rho_w = 1025.0 # Seawater density
        rho_i = 900.0  # Sea ice density
        
        self.submerged_draft = (rho_i * self.freeboard_height) / (rho_w - rho_i)
        self.total_thickness = self.freeboard_height + self.submerged_draft


class LocalPerception:
    """
    Simulates Local Exteroception (Radar, LiDAR, Sonar).
    Uses ground truth environment obstacles and generates noisy sensor readings.
    """
    def __init__(self):
        # Ground truth obstacles in the environment
        self.obstacles = [
            Obstacle(x=500.0, y=200.0, rcs=50.0, freeboard_height=1.5),
            Obstacle(x=1000.0, y=-300.0, rcs=100.0, freeboard_height=2.5),
            Obstacle(x=200.0, y=50.0, rcs=20.0, freeboard_height=0.5)
        ]
        
    def generate_radar_targets(self, vessel_x, vessel_y, vessel_heading):
        """
        Generates simulated X-Band radar targets.
        Calculates range/bearing to true targets, adds Gaussian noise,
        and adds Poisson distributed false positives (clutter).
        """
        targets = []
        
        # 1. Process true obstacles
        for obs in self.obstacles:
            dx = obs.x - vessel_x
            dy = obs.y - vessel_y
            
            true_range = math.sqrt(dx**2 + dy**2)
            if true_range > 5000: # Max radar range 5km
                continue
                
            true_bearing = math.atan2(dy, dx) - vessel_heading
            
            # Normalize bearing to [-pi, pi]
            true_bearing = (true_bearing + math.pi) % (2 * math.pi) - math.pi
            
            # Add Gaussian noise
            noisy_range = true_range + np.random.normal(0, 5.0) # 5m std dev
            noisy_bearing = true_bearing + np.random.normal(0, 0.02) # ~1 degree std dev
            
            targets.append({
                "target_id": str(uuid.uuid4()),
                "range": noisy_range,
                "bearing": noisy_bearing,
                "rcs_estimate": obs.rcs + np.random.normal(0, 2.0),
                "is_clutter": False
            })
            
        # 2. Add Poisson distributed clutter (sea waves)
        # Expected number of clutter targets per scan
        lambda_clutter = 5 
        num_clutter = np.random.poisson(lambda_clutter)
        
        for _ in range(num_clutter):
            targets.append({
                "target_id": str(uuid.uuid4()),
                "range": np.random.uniform(100, 5000),
                "bearing": np.random.uniform(-math.pi, math.pi),
                "rcs_estimate": np.random.uniform(1.0, 5.0),
                "is_clutter": True
            })
            
        return targets

    def generate_lidar_pointcloud(self, vessel_x, vessel_y):
        """
        Generates a mocked structured point cloud for obstacles nearby.
        Returns a list of point objects with [x, y, z] relative to the vessel.
        """
        point_cloud = []
        
        for obs in self.obstacles:
            dx = obs.x - vessel_x
            dy = obs.y - vessel_y
            distance = math.sqrt(dx**2 + dy**2)
            
            # Only sense within 200m
            if distance < 200:
                # Add noise to freeboard measurement
                noisy_freeboard = obs.freeboard_height + np.random.normal(0, 0.1)
                
                # We mock a 'point cloud' by just emitting the top center point of the obstacle
                # In a real system this would be thousands of points.
                point_cloud.append({
                    "x": dx, # Relative x
                    "y": dy, # Relative y
                    "z": noisy_freeboard,
                    "estimated_draft": (900.0 * noisy_freeboard) / (1025.0 - 900.0) # Onboard calculation
                })
                
        return point_cloud
        
    def check_collision(self, vessel_x, vessel_y, threshold=10.0):
        """Returns the obstacle if a collision occurs."""
        for obs in self.obstacles:
            dist = math.sqrt((obs.x - vessel_x)**2 + (obs.y - vessel_y)**2)
            if dist < threshold:
                return obs
        return None
