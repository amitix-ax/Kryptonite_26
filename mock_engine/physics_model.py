import math
import numpy as np

class PhysicsEngine:
    """
    Core 3-DOF Kinematic Physics Engine.
    Simulates vessel movement, aerodynamic drag, and ice resistance.
    """
    def __init__(self, mass=15000000.0): # 15,000 metric tons
        self.mass = mass
        
        # State Vector: x, y, yaw (theta)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Derivative Vector: surge (u), sway (v), yaw_rate (r)
        self.u = 0.0
        self.v = 0.0
        self.r = 0.0
        
        # Vessel characteristics
        self.A_T = 500.0 # Transverse projected area for wind (m^2)
        self.C_d_air = 0.8
        self.rho_a = 1.225
        self.beam = 30.0 # Ship width (m)
        self.draft = 10.0 # Ship draft (m)
        
        # Thrust state
        self.current_thrust_kN = 0.0
        
        # Physics state outputs for mock sensors
        self.last_crushing_resistance_N = 0.0

    def calculate_wind_drag(self, env_data):
        """Calculates aerodynamic drag force."""
        wind_u = env_data["wind_u"]
        wind_v = env_data["wind_v"]
        
        # Relative wind vector (simplified, ignoring ship velocity for wind calculation as wind is usually much faster)
        wind_speed_sq = wind_u**2 + wind_v**2
        
        # Force: F = 0.5 * rho * C_d * A * V^2
        f_wind = 0.5 * self.rho_a * self.C_d_air * self.A_T * wind_speed_sq
        
        # Direction of wind force (simplified, assuming wind pushes in its direction)
        wind_angle = math.atan2(wind_v, wind_u)
        
        f_wind_x = f_wind * math.cos(wind_angle)
        f_wind_y = f_wind * math.sin(wind_angle)
        
        return f_wind_x, f_wind_y

    def calculate_lindqvist_resistance(self, ice_thickness, ice_concentration, speed):
        """
        Calculates total ice resistance based on a simplified Lindqvist Formula.
        R_ice = (R_c + R_b) * (1 + 1.4 * v / sqrt(g*H)) + R_s
        """
        if ice_thickness <= 0.01 or ice_concentration < 0.1:
            self.last_crushing_resistance_N = 0.0
            return 0.0

        g = 9.81
        rho_i = 900.0
        rho_w = 1025.0
        
        # Flexural strength of ice (Pa)
        sigma_f = 500e3 
        
        # Simplified components based on Lindqvist parameters
        # R_c (Crushing) - Proportional to thickness and beam
        R_c = 0.5 * sigma_f * self.beam * ice_thickness
        self.last_crushing_resistance_N = R_c
        
        # R_b (Bending) 
        R_b = (27.0/64.0) * sigma_f * self.beam * (ice_thickness**1.5) * math.sqrt(g * (rho_w - rho_i) / rho_w)
        
        # R_s (Submersion) - Frictional drag from submerged ice blocks
        R_s = (rho_w - rho_i) * g * ice_thickness * self.beam * (self.draft * 0.5)
        
        # Velocity dependence term
        v_term = 1.0
        if speed > 0 and ice_thickness > 0:
             v_term = 1.0 + 1.4 * (speed / math.sqrt(g * ice_thickness))
             
        R_ice_total = (R_c + R_b) * v_term + R_s
        
        # Scale by concentration (simplified heuristic)
        return R_ice_total * ice_concentration

    def step(self, dt, env_data, local_perception, commanded_thrust_kN, commanded_rudder_angle):
        """
        Advances the physics simulation by dt seconds.
        """
        # Convert commanded thrust to Newtons
        thrust_N = commanded_thrust_kN * 1000.0
        self.current_thrust_kN = commanded_thrust_kN
        
        # Get environmental conditions at current location
        ice_conc = env_data["ice_concentration"]
        
        # Check collision with discrete obstacles (icebergs)
        obstacle = local_perception.check_collision(self.x, self.y)
        ice_thickness = 0.0
        if obstacle:
             ice_thickness = obstacle.total_thickness
        elif ice_conc > 0.4:
             # Level ice thickness based on concentration if no specific iceberg
             ice_thickness = 1.0 * ice_conc 

        speed = math.sqrt(self.u**2 + self.v**2)
        
        # 1. Ice Resistance
        r_ice = self.calculate_lindqvist_resistance(ice_thickness, ice_conc, speed)
        
        # 2. Hydrodynamic Drag (Water)
        c_water = 0.01
        r_water = 0.5 * 1025.0 * c_water * (self.beam * self.draft) * speed**2
        
        # Total resistance opposes motion
        total_resistance = r_ice + r_water
        
        # 3. Wind force (Leeway)
        f_wind_x, f_wind_y = self.calculate_wind_drag(env_data)
        
        # Resolve forces in ship's local frame (Surge/Sway)
        # Assuming wind is in global frame, rotate to local
        f_wind_surge = f_wind_x * math.cos(self.theta) + f_wind_y * math.sin(self.theta)
        f_wind_sway = -f_wind_x * math.sin(self.theta) + f_wind_y * math.cos(self.theta)
        
        # Surge Equation (Forward motion)
        # Thrust is forward. Resistance is backward.
        net_force_surge = thrust_N - total_resistance + f_wind_surge
        
        # Acceleration
        a_surge = net_force_surge / self.mass
        
        # Sway Equation (Lateral drift)
        # Resisted by high lateral drag of the hull
        r_sway_drag = 0.5 * 1025.0 * 1.0 * (100.0 * self.draft) * (self.v * abs(self.v)) # Huge drag coefficient laterally
        net_force_sway = f_wind_sway - r_sway_drag
        a_sway = net_force_sway / self.mass
        
        # Yaw Equation (Rotation)
        # Simplified: rudder commands yaw rate directly, resisted by ice/water
        yaw_moment = commanded_rudder_angle * 1e6 # Arbitrary steering power
        moment_of_inertia = self.mass * (100**2) / 12.0 # Approximation for a box
        yaw_drag = 1e8 * self.r * abs(self.r) # Rotational drag
        net_moment = yaw_moment - yaw_drag
        a_yaw = net_moment / moment_of_inertia
        
        # Update velocities (Euler integration)
        self.u += a_surge * dt
        self.v += a_sway * dt
        self.r += a_yaw * dt
        
        # If beset by ice (resistance > thrust and speed is very low), zero out forward velocity
        if self.u < 0.1 and thrust_N < r_ice:
             self.u = 0.0
             a_surge = 0.0 # Sudden stop
        
        # Update positions
        # Global x/y change depends on local surge(u) and sway(v) and heading(theta)
        dx = self.u * math.cos(self.theta) - self.v * math.sin(self.theta)
        dy = self.u * math.sin(self.theta) + self.v * math.cos(self.theta)
        
        self.x += dx * dt
        self.y += dy * dt
        self.theta += self.r * dt
        
        # Normalize theta
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        
        # Return the true state
        return {
            "x": self.x, "y": self.y, "theta": self.theta,
            "u": self.u, "v": self.v, "r": self.r,
            "a_surge": a_surge, "a_sway": a_sway
        }
