import rospy
import numpy as np
from geometry_msgs.msg import Twist, Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64, Header
import sys
import os

# Add package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from drone_rl_navigation.environment import DroneEnvironment
from drone_rl_navigation.models import TD3Agent

class RLController:
    """ROS node for RL-based drone navigation"""
    
    def __init__(self):
        rospy.init_node('rl_controller', anonymous=True)
        
        # Parameters
        self.rate = rospy.Rate(10)  # 10 Hz
        self.model_path = rospy.get_param('~model_path', 'models/trained_agent.npy')
        self.use_trained = rospy.get_param('~use_trained', False)
        
        # Initialize environment and agent
        self.env = DroneEnvironment()
        self.agent = TD3Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        # Load trained model if specified
        if self.use_trained and os.path.exists(self.model_path):
            try:
                self.agent.load_model(self.model_path)
                rospy.loginfo(f"Loaded trained model from {self.model_path}")
            except Exception as e:
                rospy.logwarn(f"Failed to load model: {e}")
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/drone/cmd_vel', Twist, queue_size=10)
        self.position_pub = rospy.Publisher('/drone/position', Point, queue_size=10)
        self.path_pub = rospy.Publisher('/drone/path', Path, queue_size=10)
        self.reward_pub = rospy.Publisher('/drone/reward', Float64, queue_size=10)
        
        # Subscribers
        self.goal_sub = rospy.Subscriber('/drone/goal', Point, self.goal_callback)
        
        # State
        self.current_state = self.env.reset()
        self.path_msg = Path()
        self.path_msg.header.frame_id = "world"
        self.episode_reward = 0
        self.step_count = 0
        
        rospy.loginfo("RL Controller initialized")
    
    def goal_callback(self, msg):
        """Handle new goal position"""
        self.env.goal = np.array([msg.x, msg.y, msg.z])
        self.current_state = self.env.reset()
        self.path_msg.poses = []
        self.episode_reward = 0
        self.step_count = 0
        rospy.loginfo(f"New goal set: [{msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}]")
    
    def publish_command(self, action):
        """Publish velocity command"""
        cmd = Twist()
        cmd.linear.x = action[0] * 2.0  # Scale to reasonable velocities
        cmd.linear.y = action[1] * 2.0
        cmd.linear.z = action[2] * 2.0
        self.cmd_vel_pub.publish(cmd)
    
    def publish_position(self):
        """Publish current position"""
        pos = Point()
        pos.x = self.env.position[0]
        pos.y = self.env.position[1]
        pos.z = self.env.position[2]
        self.position_pub.publish(pos)
        
        # Add to path
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "world"
        pose.pose.position = pos
        self.path_msg.poses.append(pose)
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_pub.publish(self.path_msg)
    
    def run(self):
        """Main control loop"""
        rospy.loginfo("Starting RL controller loop")
        
        while not rospy.is_shutdown():
            # Select action using policy
            action = self.agent.select_action(self.current_state, noise=0.05)
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Publish commands and state
            self.publish_command(action)
            self.publish_position()
            
            # Publish reward
            self.episode_reward += reward
            self.reward_pub.publish(Float64(self.episode_reward))
            
            # Log progress
            self.step_count += 1
            if self.step_count % 50 == 0:
                dist = info['goal_distance']
                vel = info['velocity']
                rospy.loginfo(f"Step {self.step_count}: Distance={dist:.2f}m, "
                            f"Velocity={vel:.2f}m/s, Reward={self.episode_reward:.2f}")
            
            # Reset if episode done
            if done:
                rospy.loginfo(f"Episode finished. Total reward: {self.episode_reward:.2f}")
                self.current_state = self.env.reset()
                self.path_msg.poses = []
                self.episode_reward = 0
                self.step_count = 0
            else:
                self.current_state = next_state
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = RLController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
