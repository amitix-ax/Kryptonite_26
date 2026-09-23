import numpy as np
import time

class MacroEnvironment:
    """
    Simulates macro-environmental data (Open Web Data).
    Generates synthetic 2D grids of ice concentration, wind, and ocean currents.
    """
    def __init__(self, grid_size=(100, 100), resolution_km=1.0):
        self.grid_size = grid_size
        self.resolution_km = resolution_km
        self.last_update_time = time.time()
        self.update_interval = 21600  # 6 hours in seconds (satellite latency)
        
        # Initialize grids
        self.ice_concentration = np.zeros(grid_size)
        self.ice_drift_x = np.zeros(grid_size)
        self.ice_drift_y = np.zeros(grid_size)
        self.wind_u = np.zeros(grid_size)
        self.wind_v = np.zeros(grid_size)
        self.current_u = np.zeros(grid_size)
        self.current_v = np.zeros(grid_size)
        
        self.generate_synthetic_environment()

    def generate_synthetic_environment(self):
        """Generates a base synthetic environment with perlin-like noise (simplified)."""
        # Simplistic generation of ice concentration with a gradient and some random clusters
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Create an ice edge roughly in the middle
                base_concentration = 1.0 if j > self.grid_size[1] // 2 else 0.0
                noise = np.random.normal(0, 0.2)
                self.ice_concentration[i, j] = np.clip(base_concentration + noise, 0.0, 1.0)
                
                # Wind and current vectors (general drift)
                self.wind_u[i, j] = np.random.normal(5.0, 1.0) # Prevailing wind
                self.wind_v[i, j] = np.random.normal(2.0, 0.5)
                self.current_u[i, j] = np.random.normal(0.5, 0.1)
                self.current_v[i, j] = np.random.normal(0.1, 0.05)
                
                # Ice drift is a function of wind and current
                self.ice_drift_x[i, j] = 0.02 * self.wind_u[i, j] + 0.8 * self.current_u[i, j]
                self.ice_drift_y[i, j] = 0.02 * self.wind_v[i, j] + 0.8 * self.current_v[i, j]

    def _apply_latency_blur(self):
        """Simulate satellite data staleness by applying a slight blur/noise over time."""
        # In a real scenario, this would interpolate between old and new satellite passes.
        # Here we just add some random noise to simulate uncertainty of old data.
        noise_level = 0.05
        self.ice_concentration += np.random.normal(0, noise_level, self.grid_size)
        self.ice_concentration = np.clip(self.ice_concentration, 0.0, 1.0)
        
    def get_environment_at(self, x, y):
        """Returns the environmental parameters at a specific continuous x,y coordinate."""
        # Convert continuous coordinates to grid indices
        # Assuming origin (0,0) is at the center of the grid for simplicity
        grid_x = int(x / (self.resolution_km * 1000) + self.grid_size[0] / 2)
        grid_y = int(y / (self.resolution_km * 1000) + self.grid_size[1] / 2)
        
        # Boundary check
        if 0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]:
            # Check if it's time to "receive" a new satellite update
            current_time = time.time()
            if current_time - self.last_update_time > self.update_interval:
                self.generate_synthetic_environment()
                self.last_update_time = current_time
            else:
                # Apply degradation if it's stale
                # For high frequency polling, we shouldn't blur every time, but just simulate that the data we hold is stale
                pass
                
            return {
                "ice_concentration": self.ice_concentration[grid_x, grid_y],
                "ice_drift_x": self.ice_drift_x[grid_x, grid_y],
                "ice_drift_y": self.ice_drift_y[grid_x, grid_y],
                "wind_u": self.wind_u[grid_x, grid_y],
                "wind_v": self.wind_v[grid_x, grid_y],
                "current_u": self.current_u[grid_x, grid_y],
                "current_v": self.current_v[grid_x, grid_y]
            }
        else:
            # Out of bounds
            return {
                "ice_concentration": 0.0,
                "ice_drift_x": 0.0, "ice_drift_y": 0.0,
                "wind_u": 0.0, "wind_v": 0.0,
                "current_u": 0.0, "current_v": 0.0
            }
