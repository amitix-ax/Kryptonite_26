import numpy as np

class ProprioceptionMock:
    """
    Mocks proprioceptive sensors (IMU, Propulsion, Strain Gauges).
    Takes true physical states and applies sensor noise profiles.
    """
    def __init__(self):
        # IMU drift states (Random Walk)
        self.imu_drift = {
            "x": 0.0, "y": 0.0, "theta": 0.0,
            "u": 0.0, "v": 0.0, "r": 0.0
        }
        
        # Drift standard deviations per step
        self.sigma_rw = {
            "pos": 0.001,
            "angle": 0.0001,
            "vel": 0.005,
            "rate": 0.0005
        }
        
        # Ship constants for strain calculation
        self.bow_area = 50.0 # Projected contact area of bow (m^2)

    def generate_imu_telemetry(self, true_state):
        """
        Applies random walk drift to the true state to simulate an IMU.
        """
        # Update random walk drift
        self.imu_drift["x"] += np.random.normal(0, self.sigma_rw["pos"])
        self.imu_drift["y"] += np.random.normal(0, self.sigma_rw["pos"])
        self.imu_drift["theta"] += np.random.normal(0, self.sigma_rw["angle"])
        
        self.imu_drift["u"] += np.random.normal(0, self.sigma_rw["vel"])
        self.imu_drift["v"] += np.random.normal(0, self.sigma_rw["vel"])
        self.imu_drift["r"] += np.random.normal(0, self.sigma_rw["rate"])
        
        # Add high-frequency white noise to accelerations
        noisy_a_surge = true_state["a_surge"] + np.random.normal(0, 0.05)
        noisy_a_sway = true_state["a_sway"] + np.random.normal(0, 0.05)
        
        return {
            "x": true_state["x"] + self.imu_drift["x"],
            "y": true_state["y"] + self.imu_drift["y"],
            "theta": true_state["theta"] + self.imu_drift["theta"],
            "u": true_state["u"] + self.imu_drift["u"],
            "v": true_state["v"] + self.imu_drift["v"],
            "r": true_state["r"] + self.imu_drift["r"],
            "a_surge": noisy_a_surge,
            "a_sway": noisy_a_sway
        }

    def generate_azipod_telemetry(self, commanded_thrust_kN):
        """
        Simulates Azipod response.
        Adds noise to torque and RPM based on requested thrust.
        """
        # Simplified linear relation for MVP: Thrust roughly proportional to RPM^2
        # T = K * rpm^2 => rpm = sqrt(T / K)
        K = 0.5 
        base_rpm = 0.0
        if commanded_thrust_kN > 0:
            base_rpm = np.sqrt(commanded_thrust_kN / K)
        elif commanded_thrust_kN < 0:
            base_rpm = -np.sqrt(-commanded_thrust_kN / K)
            
        rpm_noise = np.random.normal(0, 2.0)
        noisy_rpm = int(base_rpm + rpm_noise)
        
        # Torque relates to RPM and resistance. We mock it with noise.
        base_torque = commanded_thrust_kN * 1.5 
        noisy_torque = base_torque + np.random.normal(0, 50.0)
        
        noisy_thrust = commanded_thrust_kN + np.random.normal(0, 20.0)
        
        return {
            "shaft_rpm": noisy_rpm,
            "thrust_kN": noisy_thrust,
            "torque_kNm": noisy_torque
        }

    def generate_strain_gauge_telemetry(self, crushing_resistance_N):
        """
        Simulates Fiber Bragg Grating (FBG) strain gauges on the hull.
        """
        # Stress = Force / Area (Pascals)
        # Convert to MPa
        stress_MPa = (crushing_resistance_N / self.bow_area) / 1e6
        
        # Add engine vibration and ice milling noise (high frequency white noise)
        noise = np.random.normal(0, 0.5) # 0.5 MPa variance
        
        noisy_stress = max(0.0, stress_MPa + noise)
        
        return {
            "hull_stress_MPa": noisy_stress
        }
