import numpy as np
import rospy
from geometry_msgs.msg import Point, Twist, Vector3
from std_msgs.msg import Float64
import random

class DroneEnvironment:
    """Custom RL environment for drone navigation"""
    
    def __init__(self, goal_position=None, max_steps=1000):
        self.max_steps = max_steps
        self.current_step = 0
        
        # State space: [x, y, z, vx, vy, vz, goal_x, goal_y, goal_z]
        self.state_dim = 9
        # Action space: [thrust_x, thrust_y, thrust_z]
        self.action_dim = 3
        
        # Environment boundaries
        self.bounds = {
            'x': (-10, 10),
            'y': (-10, 10),
            'z': (0, 10)
        }
        
        # Drone state
        self.position = np.array([0.0, 0.0, 1.0])
        self.velocity = np.zeros(3)
        self.goal = goal_position if goal_position else np.array([8.0, 8.0, 5.0])
        
        # Physical parameters
        self.mass = 1.5  # kg
        self.drag_coeff = 0.1
        self.dt = 0.1  # time step
        self.max_velocity = 5.0  # m/s
        self.max_thrust = 15.0  # N
        
        # Obstacles (spherical)
        self.obstacles = self._generate_obstacles()
        
        # Reward parameters
        self.goal_threshold = 0.5
        self.collision_penalty = -100
        self.goal_reward = 100
        self.step_penalty = -0.1
        
    def _generate_obstacles(self, num_obstacles=5):
        """Generate random obstacles in the environment"""
        obstacles = []
        for _ in range(num_obstacles):
            obs = {
                'center': np.array([
                    random.uniform(self.bounds['x'][0], self.bounds['x'][1]),
                    random.uniform(self.bounds['y'][0], self.bounds['y'][1]),
                    random.uniform(self.bounds['z'][0], self.bounds['z'][1])
                ]),
                'radius': random.uniform(0.5, 1.5)
            }
            obstacles.append(obs)
        return obstacles
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.position = np.array([
            random.uniform(-2, 2),
            random.uniform(-2, 2),
            random.uniform(1, 2)
        ])
        self.velocity = np.zeros(3)
        self.obstacles = self._generate_obstacles()
        return self._get_state()
    
    def _get_state(self):
        """Get current state observation"""
        relative_goal = self.goal - self.position
        state = np.concatenate([
            self.position,
            self.velocity,
            relative_goal
        ])
        return state
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.current_step += 1
        
        # Clip actions
        action = np.clip(action, -1, 1) * self.max_thrust
        
        # Physics update
        acceleration = action / self.mass - self.drag_coeff * self.velocity
        acceleration[2] -= 9.81  # gravity
        
        self.velocity += acceleration * self.dt
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
        
        self.position += self.velocity * self.dt
        
        # Calculate reward and check termination
        reward = self.step_penalty
        done = False
        
        # Check goal reached
        dist_to_goal = np.linalg.norm(self.position - self.goal)
        if dist_to_goal < self.goal_threshold:
            reward += self.goal_reward
            done = True
            
        # Check collision with obstacles
        for obs in self.obstacles:
            dist_to_obs = np.linalg.norm(self.position - obs['center'])
            if dist_to_obs < obs['radius']:
                reward += self.collision_penalty
                done = True
                break
        
        # Check out of bounds
        if (self.position[0] < self.bounds['x'][0] or self.position[0] > self.bounds['x'][1] or
            self.position[1] < self.bounds['y'][0] or self.position[1] > self.bounds['y'][1] or
            self.position[2] < self.bounds['z'][0] or self.position[2] > self.bounds['z'][1]):
            reward += self.collision_penalty
            done = True
        
        # Max steps reached
        if self.current_step >= self.max_steps:
            done = True
        
        # Distance-based reward shaping
        reward -= dist_to_goal * 0.01
        
        next_state = self._get_state()
        info = {
            'position': self.position.copy(),
            'goal_distance': dist_to_goal,
            'velocity': np.linalg.norm(self.velocity)
        }
        
        return next_state, reward, done, info
    
    def get_obstacles(self):
        """Return obstacle information for visualization"""
        return self.obstacles
    
    def get_goal(self):
        """Return goal position"""
        return self.goal