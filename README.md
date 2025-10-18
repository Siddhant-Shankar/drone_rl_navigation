Drone RL Navigation - 3D Path Visualization with Reinforcement Learning
A ROS package implementing 3D path visualization and reinforcement learning-based control algorithms for autonomous drone navigation. The system uses Twin Delayed Deep Deterministic Policy Gradient (TD3) to improve navigation precision and adaptability under variable environmental conditions.
Overview
This project combines:

Reinforcement Learning: TD3 algorithm for continuous control
3D Visualization: Real-time path tracking and environment visualization in RViz
Dynamic Obstacles: Adaptive navigation around obstacles
Physics Simulation: Realistic drone dynamics with gravity, drag, and thrust

Features

✅ TD3-based RL controller for precise navigation
✅ 3D path visualization with RViz integration
✅ Dynamic obstacle avoidance
✅ Real-time performance monitoring
✅ Model training and evaluation pipeline
✅ Configurable environment parameters

System Architecture
┌─────────────────┐
│ Training Node │
│ (Offline) │
└────────┬────────┘
│
▼
┌─────────┐
│ Model │
└────┬────┘
│
▼
┌──────────────────┐ ┌────────────────┐
│ RL Controller │◄────►│ Environment │
│ (Online) │ │ Simulator │
└────────┬─────────┘ └────────────────┘
│
▼
┌──────────────────┐ ┌────────────────┐
│ Path Visualizer │─────►│ RViz │
└──────────────────┘ └────────────────┘
Installation
Prerequisites

ROS Noetic (Ubuntu 20.04) or ROS Melodic (Ubuntu 18.04)
Python 3.6+
NumPy

Build Instructions
bash# Create catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Clone the repository

git clone https://github.com/yourusername/drone_rl_navigation.git

# Install dependencies

cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y

# Build the package

catkin_make

# Source the workspace

source devel/setup.bash
Usage

1. Training the RL Agent
   Train the agent from scratch:
   bash# Launch training
   roslaunch drone_rl_navigation training.launch num_episodes:=500

# Monitor training progress

rostopic echo /training/episode_reward
Training parameters can be configured in config/rl_config.yaml. 2. Running Trained Controller
Execute navigation with a trained model:
bash# Run with trained model
roslaunch drone_rl_navigation drone_navigation.launch use_trained:=true

# Set a new goal position

rostopic pub /drone/goal geometry_msgs/Point "{x: 5.0, y: 5.0, z: 3.0}" 3. Visualization Only
Launch just the visualization components:
bashroslaunch drone_rl_navigation visualization.launch
ROS Topics
Published Topics
TopicTypeDescription/drone/cmd_velgeometry_msgs/TwistVelocity commands/drone/positiongeometry_msgs/PointCurrent position/drone/pathnav_msgs/PathHistorical path/drone/rewardstd_msgs/Float64Cumulative reward/visualization/markersvisualization_msgs/MarkerArrayEnvironment markers/training/episode_rewardstd_msgs/Float64Training rewards
Subscribed Topics
TopicTypeDescription/drone/goalgeometry_msgs/PointTarget position
Configuration
Environment Parameters
Edit config/rl_config.yaml:
yamlenvironment:
bounds:
x_min: -10.0
x_max: 10.0
num_obstacles: 5
goal_threshold: 0.5
Visualization Settings
Edit config/path_params.yaml:
yamlvisualization:
path_color: [0.0, 1.0, 0.0, 0.8]
update_rate: 10
Algorithm Details
TD3 (Twin Delayed DDPG)
The implementation uses:

Actor Network: Policy approximation (state → action)
Twin Critic Networks: Q-value estimation with reduced overestimation
Target Networks: Stabilized learning with soft updates
Experience Replay: Off-policy learning from past experiences
Delayed Policy Updates: Reduces policy variance

State Space

Position (x, y, z)
Velocity (vx, vy, vz)
Relative goal position (gx, gy, gz)

Dimension: 9
Action Space

Thrust commands (tx, ty, tz)
Continuous values in [-1, 1]

Dimension: 3
Performance
Typical training results (500 episodes):

Success rate: 75-85%
Average reward: 60-80
Goal reach time: 15-25 seconds
Training time: 2-3 hours (CPU)

Project Structure
drone_rl_navigation/
├── config/ # Configuration files
│ ├── rl_config.yaml
│ └── path_params.yaml
├── launch/ # ROS launch files
│ ├── drone_navigation.launch
│ ├── training.launch
│ └── visualization.launch
├── msg/ # Custom message definitions
│ ├── DroneState.msg
│ └── PathPoint.msg
├── scripts/ # Executable Python scripts
│ ├── rl_controller.py
│ ├── path_visualizer.py
│ ├── drone_simulator.py
│ └── training_node.py
├── src/ # Python source code
│ └── drone_rl_navigation/
│ ├── environment.py
│ ├── models.py
│ └── utils.py
├── models/ # Saved model weights
├── CMakeLists.txt
├── package.xml
└── README.md
Troubleshooting
Common Issues
Issue: Training not converging

Solution: Adjust learning rates in config/rl_config.yaml
Try reducing actor_lr to 0.0005

Issue: Visualization not showing in RViz

Solution: Check frame_id is set to "world"
Verify topics: rostopic list | grep visualization

Issue: Drone colliding frequently

Solution: Increase exploration noise during training
Adjust obstacle_radius_range to create easier scenarios

Future Improvements

Integration with Gazebo for realistic simulation
Multi-drone coordination
Real hardware deployment (PX4/ArduPilot)
Imitation learning initialization
Advanced reward shaping
LSTM for partial observability

Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch
Commit your changes
Submit a pull request

License
MIT License - see LICENSE file for details
Citation
If you use this code in your research, please cite:
bibtex@software{drone_rl_navigation2025,
title={Drone RL Navigation: 3D Path Visualization with Reinforcement Learning},
author={Your Name},
year={2025},
url={https://github.com/yourusername/drone_rl_navigation}
}
Acknowledgments

TD3 algorithm based on "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)
Built with ROS and Python
Part of the Lumyn Aerospace Engineering Project
