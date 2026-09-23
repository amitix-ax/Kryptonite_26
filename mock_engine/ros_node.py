import time
import json
import threading

# Import mock engine components
from macro_env import MacroEnvironment
from local_perception import LocalPerception
from physics_model import PhysicsEngine
from proprioception import ProprioceptionMock

# Try to import ROS2, otherwise fallback to mock publisher
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String # Simplified: using JSON strings for MVP instead of complex custom msg types
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("WARNING: rclpy not found. Running with Mock Publisher (Console Output).")

class MockPublisher:
    """Fallback publisher if ROS2 is not installed."""
    def publish(self, topic, data):
        # We only print occasionally to not flood the console in mock mode
        if np.random.random() < 0.01:
             print(f"[{topic}] {json.dumps(data)[:100]}...")

if ROS2_AVAILABLE:
    class DigitalTwinNode(Node):
        def __init__(self):
            super().__init__('kryptonite_digital_twin')
            self.pub_macro = self.create_publisher(String, '/environment/macro', 10)
            self.pub_radar = self.create_publisher(String, '/sensors/radar/targets', 10)
            self.pub_lidar = self.create_publisher(String, '/sensors/lidar/pointcloud', 10)
            self.pub_imu = self.create_publisher(String, '/vessel/imu', 10)
            self.pub_prop = self.create_publisher(String, '/vessel/propulsion', 10)
            self.pub_strain = self.create_publisher(String, '/vessel/strain', 10)
            
        def publish_data(self, pub, data):
            msg = String()
            msg.data = json.dumps(data)
            pub.publish(msg)

class DigitalTwinRunner:
    def __init__(self):
        self.macro_env = MacroEnvironment()
        self.local_perception = LocalPerception()
        self.physics = PhysicsEngine()
        self.proprioception = ProprioceptionMock()
        
        self.dt = 0.01 # 100Hz
        
        if ROS2_AVAILABLE:
            rclpy.init()
            self.node = DigitalTwinNode()
        else:
            self.node = None
            self.mock_pub = MockPublisher()
            
        # Mock DSS commands (in reality, DSS sends these to the ship)
        self.commanded_thrust_kN = 5000.0 # 5 MN thrust
        self.commanded_rudder = 0.0

    def publish(self, topic_name, data):
        if ROS2_AVAILABLE:
            if topic_name == '/environment/macro': self.node.publish_data(self.node.pub_macro, data)
            elif topic_name == '/sensors/radar/targets': self.node.publish_data(self.node.pub_radar, data)
            elif topic_name == '/sensors/lidar/pointcloud': self.node.publish_data(self.node.pub_lidar, data)
            elif topic_name == '/vessel/imu': self.node.publish_data(self.node.pub_imu, data)
            elif topic_name == '/vessel/propulsion': self.node.publish_data(self.node.pub_prop, data)
            elif topic_name == '/vessel/strain': self.node.publish_data(self.node.pub_strain, data)
        else:
            self.mock_pub.publish(topic_name, data)

    def run_loop(self):
        print("Starting Digital Twin Mock Engine...")
        last_time = time.time()
        
        step_count = 0
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - last_time
                
                if elapsed >= self.dt:
                    last_time = current_time
                    
                    # 1. Get Environmental Data
                    env_data = self.macro_env.get_environment_at(self.physics.x, self.physics.y)
                    
                    # 2. Step Physics
                    true_state = self.physics.step(
                        self.dt, 
                        env_data, 
                        self.local_perception, 
                        self.commanded_thrust_kN, 
                        self.commanded_rudder
                    )
                    
                    # 3. Generate Mock Telemetry
                    # Publish IMU (100Hz)
                    imu_data = self.proprioception.generate_imu_telemetry(true_state)
                    self.publish('/vessel/imu', imu_data)
                    
                    # Publish Strain (100Hz)
                    strain_data = self.proprioception.generate_strain_gauge_telemetry(self.physics.last_crushing_resistance_N)
                    self.publish('/vessel/strain', strain_data)
                    
                    # Lower frequency publishes (e.g. 10Hz or 1Hz)
                    if step_count % 10 == 0: # 10Hz
                        prop_data = self.proprioception.generate_azipod_telemetry(self.commanded_thrust_kN)
                        self.publish('/vessel/propulsion', prop_data)
                        
                    if step_count % 100 == 0: # 1Hz
                        radar_data = self.local_perception.generate_radar_targets(true_state["x"], true_state["y"], true_state["theta"])
                        self.publish('/sensors/radar/targets', radar_data)
                        
                        lidar_data = self.local_perception.generate_lidar_pointcloud(true_state["x"], true_state["y"])
                        self.publish('/sensors/lidar/pointcloud', lidar_data)
                        
                        self.publish('/environment/macro', env_data)
                        
                        # Debug output if no ROS2
                        if not ROS2_AVAILABLE:
                             print(f"Physics State -> Speed: {np.sqrt(true_state['u']**2 + true_state['v']**2):.2f} m/s, X: {true_state['x']: .1f}, Y: {true_state['y']:.1f}, Ice Res: {self.physics.last_crushing_resistance_N/1000:.1f} kN")
                        
                    step_count += 1
                    
        except KeyboardInterrupt:
            print("Shutting down Mock Engine.")
            if ROS2_AVAILABLE:
                self.node.destroy_node()
                rclpy.shutdown()

import numpy as np # Added here since it's used in the fallback mock pub
if __name__ == '__main__':
    runner = DigitalTwinRunner()
    runner.run_loop()
