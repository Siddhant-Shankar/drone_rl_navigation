import rospy
import numpy as np
from geometry_msgs.msg import Twist, Point, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from drone_rl_navigation.utils import quaternion_from_euler

class DroneSimulator:
    """Simple physics-based drone simulator"""
    
    def __init__(self):
        rospy.init_node('drone_simulator', anonymous=True)
        
        # Simulation parameters
        self.rate = rospy.Rate(50)  # 50 Hz
        self.dt = 0.02  # 20ms timestep
        
        # Physical parameters
        self.mass = 1.5  # kg
        self.drag_coeff = 0.1
        self.gravity = 9.81  # m/s^2
        self.max_thrust = 20.0  # N
        
        # State variables
        self.position = np.array([0.0, 0.0, 1.0])
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.angular_velocity = np.zeros(3)
        
        # Control input
        self.cmd_velocity = np.zeros(3)
        
        # Publishers
        self.odom_pub = rospy.Publisher('/drone/odometry', Odometry, queue_size=10)
        self.pose_pub = rospy.Publisher('/drone/pose', PoseStamped, queue_size=10)
        self.imu_pub = rospy.Publisher('/drone/imu', Imu, queue_size=10)
        
        # Subscribers
        self.cmd_sub = rospy.Subscriber('/drone/cmd_vel', Twist, self.cmd_callback)
        
        rospy.loginfo("Drone Simulator initialized")
    
    def cmd_callback(self, msg):
        """Handle velocity commands"""
        self.cmd_velocity = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.linear.z
        ])
    
    def update_physics(self):
        """Update drone physics simulation"""
        # Compute thrust from commanded velocity (simplified controller)
        desired_accel = (self.cmd_velocity - self.velocity) * 2.0
        thrust = np.clip(desired_accel * self.mass, -self.max_thrust, self.max_thrust)
        
        # Compute total acceleration
        drag_force = -self.drag_coeff * self.velocity
        gravity_force = np.array([0, 0, -self.gravity * self.mass])
        
        total_force = thrust + drag_force + gravity_force
        self.acceleration = total_force / self.mass
        
        # Update velocity and position (Euler integration)
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt
        
        # Ground collision
        if self.position[2] < 0:
            self.position[2] = 0
            self.velocity[2] = max(0, self.velocity[2])
        
        # Simple orientation from velocity
        if np.linalg.norm(self.velocity[:2]) > 0.1:
            self.orientation[2] = np.arctan2(self.velocity[1], self.velocity[0])
    
    def publish_odometry(self):
        """Publish odometry message"""
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world"
        odom.child_frame_id = "drone_base"
        
        # Position
        odom.pose.pose.position.x = self.position[0]
        odom.pose.pose.position.y = self.position[1]
        odom.pose.pose.position.z = self.position[2]
        
        # Orientation
        q = quaternion_from_euler(
            self.orientation[0],
            self.orientation[1],
            self.orientation[2]
        )
        odom.pose.pose.orientation = q
        
        # Velocity
        odom.twist.twist.linear.x = self.velocity[0]
        odom.twist.twist.linear.y = self.velocity[1]
        odom.twist.twist.linear.z = self.velocity[2]
        
        self.odom_pub.publish(odom)
    
    def publish_pose(self):
        """Publish pose message"""
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "world"
        
        pose.pose.position.x = self.position[0]
        pose.pose.position.y = self.position[1]
        pose.pose.position.z = self.position[2]
        
        q = quaternion_from_euler(
            self.orientation[0],
            self.orientation[1],
            self.orientation[2]
        )
        pose.pose.orientation = q
        
        self.pose_pub.publish(pose)
    
    def publish_imu(self):
        """Publish IMU message"""
        imu = Imu()
        imu.header.stamp = rospy.Time.now()
        imu.header.frame_id = "drone_imu"
        
        # Orientation
        q = quaternion_from_euler(
            self.orientation[0],
            self.orientation[1],
            self.orientation[2]
        )
        imu.orientation = q
        
        # Angular velocity
        imu.angular_velocity.x = self.angular_velocity[0]
        imu.angular_velocity.y = self.angular_velocity[1]
        imu.angular_velocity.z = self.angular_velocity[2]
        
        # Linear acceleration
        imu.linear_acceleration.x = self.acceleration[0]
        imu.linear_acceleration.y = self.acceleration[1]
        imu.linear_acceleration.z = self.acceleration[2] + self.gravity
        
        self.imu_pub.publish(imu)
    
    def run(self):
        """Main simulation loop"""
        rospy.loginfo("Starting drone simulation")
        
        while not rospy.is_shutdown():
            # Update physics
            self.update_physics()
            
            # Publish state
            self.publish_odometry()
            self.publish_pose()
            self.publish_imu()
            
            # Log periodically
            if rospy.get_time() % 5.0 < self.dt:
                rospy.loginfo(f"Drone State - Pos: [{self.position[0]:.2f}, "
                            f"{self.position[1]:.2f}, {self.position[2]:.2f}], "
                            f"Vel: {np.linalg.norm(self.velocity):.2f} m/s")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        simulator = DroneSimulator()
        simulator.run()
    except rospy.ROSInterruptException:
        pass