import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion
import math

def distance_3d(p1, p2):
    """Calculate Euclidean distance between two 3D points"""
    if isinstance(p1, Point):
        p1 = np.array([p1.x, p1.y, p1.z])
    if isinstance(p2, Point):
        p2 = np.array([p2.x, p2.y, p2.z])
    return np.linalg.norm(p1 - p2)

def normalize_vector(v):
    """Normalize a vector"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm

def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion"""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    
    return q

def euler_from_quaternion(q):
    """Convert quaternion to Euler angles"""
    if isinstance(q, Quaternion):
        x, y, z, w = q.x, q.y, q.z, q.w
    else:
        x, y, z, w = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def clip_velocity(velocity, max_vel):
    """Clip velocity vector to maximum magnitude"""
    vel_mag = np.linalg.norm(velocity)
    if vel_mag > max_vel:
        return velocity * (max_vel / vel_mag)
    return velocity

def compute_path_length(path):
    """Compute total length of a path"""
    if len(path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(path) - 1):
        total_length += distance_3d(path[i], path[i + 1])
    
    return total_length

def smooth_path(path, window_size=5):
    """Apply moving average smoothing to path"""
    if len(path) < window_size:
        return path
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(path)):
        start = max(0, i - half_window)
        end = min(len(path), i + half_window + 1)
        
        avg_point = np.mean([np.array([p.x, p.y, p.z]) for p in path[start:end]], axis=0)
        
        p = Point()
        p.x, p.y, p.z = avg_point
        smoothed.append(p)
    
    return smoothed

def check_collision_with_sphere(point, sphere_center, sphere_radius):
    """Check if point collides with sphere"""
    dist = distance_3d(point, sphere_center)
    return dist < sphere_radius

def interpolate_path(start, end, num_points=10):
    """Interpolate path between two points"""
    if isinstance(start, Point):
        start = np.array([start.x, start.y, start.z])
    if isinstance(end, Point):
        end = np.array([end.x, end.y, end.z])
    
    path = []
    for i in range(num_points):
        alpha = i / (num_points - 1)
        point = start + alpha * (end - start)
        p = Point()
        p.x, p.y, p.z = point
        path.append(p)
    
    return path

def compute_velocity_direction(current_pos, target_pos):
    """Compute unit vector from current to target position"""
    if isinstance(current_pos, Point):
        current_pos = np.array([current_pos.x, current_pos.y, current_pos.z])
    if isinstance(target_pos, Point):
        target_pos = np.array([target_pos.x, target_pos.y, target_pos.z])
    
    direction = target_pos - current_pos
    return normalize_vector(direction)

def exponential_decay(initial_value, decay_rate, step):
    """Compute exponentially decayed value"""
    return initial_value * (decay_rate ** step)

class MovingAverage:
    """Compute moving average of values"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = []
    
    def update(self, value):
        """Add new value and return current average"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return self.get_average()
    
    def get_average(self):
        """Get current average"""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def reset(self):
        """Reset the moving average"""
        self.values = []

class Timer:
    """Simple timer utility"""
    
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
    
    def start(self):
        """Start the timer"""
        self.start_time = rospy.Time.now()
    
    def stop(self):
        """Stop the timer and return elapsed time"""
        if self.start_time is None:
            return 0.0
        
        self.elapsed_time = (rospy.Time.now() - self.start_time).to_sec()
        self.start_time = None
        return self.elapsed_time
    
    def get_elapsed(self):
        """Get elapsed time without stopping"""
        if self.start_time is None:
            return self.elapsed_time
        return (rospy.Time.now() - self.start_time).to_sec()

def save_trajectory(trajectory, filename):
    """Save trajectory to file"""
    import json
    
    data = []
    for pose in trajectory:
        data.append({
            'x': pose.pose.position.x,
            'y': pose.pose.position.y,
            'z': pose.pose.position.z,
            'timestamp': pose.header.stamp.to_sec()
        })
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    rospy.loginfo(f"Trajectory saved to {filename}")

def load_trajectory(filename):
    """Load trajectory from file"""
    import json
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    path = Path()
    path.header.frame_id = "world"
    
    for point in data:
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.header.stamp = rospy.Time(point['timestamp'])
        pose.pose.position.x = point['x']
        pose.pose.position.y = point['y']
        pose.pose.position.z = point['z']
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    
    rospy.loginfo(f"Trajectory loaded from {filename}")
    return path