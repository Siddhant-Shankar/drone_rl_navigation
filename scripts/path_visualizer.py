import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from std_msgs.msg import ColorRGBA
import numpy as np

class PathVisualizer:
    """Visualize drone path and environment in 3D"""
    
    def __init__(self):
        rospy.init_node('path_visualizer', anonymous=True)
        
        # Publishers
        self.marker_pub = rospy.Publisher('/visualization/markers', MarkerArray, queue_size=10)
        self.path_marker_pub = rospy.Publisher('/visualization/path_markers', Marker, queue_size=10)
        
        # Subscribers
        self.path_sub = rospy.Subscriber('/drone/path', Path, self.path_callback)
        self.position_sub = rospy.Subscriber('/drone/position', Point, self.position_callback)
        self.goal_sub = rospy.Subscriber('/drone/goal', Point, self.goal_callback)
        
        # State
        self.current_position = None
        self.goal_position = Point(8.0, 8.0, 5.0)
        self.obstacles = self._generate_obstacles()
        
        # Timer for periodic visualization
        self.timer = rospy.Timer(rospy.Duration(0.1), self.visualize_environment)
        
        rospy.loginfo("Path Visualizer initialized")
    
    def _generate_obstacles(self):
        """Generate obstacle positions (matching environment)"""
        obstacles = []
        np.random.seed(42)  # For reproducibility
        for i in range(5):
            obs = {
                'id': i,
                'center': Point(
                    x=np.random.uniform(-10, 10),
                    y=np.random.uniform(-10, 10),
                    z=np.random.uniform(0, 10)
                ),
                'radius': np.random.uniform(0.5, 1.5)
            }
            obstacles.append(obs)
        return obstacles
    
    def path_callback(self, msg):
        """Handle path updates"""
        if len(msg.poses) == 0:
            return
        
        # Visualize path as line strip
        marker = Marker()
        marker.header = msg.header
        marker.ns = "drone_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.8)  # Green
        marker.pose.orientation.w = 1.0
        
        for pose in msg.poses:
            marker.points.append(pose.pose.position)
        
        self.path_marker_pub.publish(marker)
    
    def position_callback(self, msg):
        """Handle position updates"""
        self.current_position = msg
    
    def goal_callback(self, msg):
        """Handle goal updates"""
        self.goal_position = msg
        rospy.loginfo(f"Goal visualization updated: [{msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}]")
    
    def visualize_environment(self, event):
        """Visualize obstacles, goal, and drone"""
        marker_array = MarkerArray()
        
        # Visualize obstacles as spheres
        for obs in self.obstacles:
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = obs['id']
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position = obs['center']
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = obs['radius'] * 2
            marker.scale.y = obs['radius'] * 2
            marker.scale.z = obs['radius'] * 2
            
            marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.5)  # Red, semi-transparent
            
            marker_array.markers.append(marker)
        
        # Visualize goal as a star/sphere
        goal_marker = Marker()
        goal_marker.header.frame_id = "world"
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.ns = "goal"
        goal_marker.id = 100
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        
        goal_marker.pose.position = self.goal_position
        goal_marker.pose.orientation.w = 1.0
        
        goal_marker.scale.x = 0.6
        goal_marker.scale.y = 0.6
        goal_marker.scale.z = 0.6
        
        goal_marker.color = ColorRGBA(1.0, 1.0, 0.0, 0.9)  # Yellow
        
        marker_array.markers.append(goal_marker)
        
        # Visualize current drone position
        if self.current_position:
            drone_marker = Marker()
            drone_marker.header.frame_id = "world"
            drone_marker.header.stamp = rospy.Time.now()
            drone_marker.ns = "drone"
            drone_marker.id = 101
            drone_marker.type = Marker.CUBE
            drone_marker.action = Marker.ADD
            
            drone_marker.pose.position = self.current_position
            drone_marker.pose.orientation.w = 1.0
            
            drone_marker.scale.x = 0.3
            drone_marker.scale.y = 0.3
            drone_marker.scale.z = 0.1
            
            drone_marker.color = ColorRGBA(0.0, 0.0, 1.0, 1.0)  # Blue
            
            marker_array.markers.append(drone_marker)
        
        # Visualize boundary box
        boundary_marker = Marker()
        boundary_marker.header.frame_id = "world"
        boundary_marker.header.stamp = rospy.Time.now()
        boundary_marker.ns = "boundary"
        boundary_marker.id = 102
        boundary_marker.type = Marker.LINE_LIST
        boundary_marker.action = Marker.ADD
        
        boundary_marker.scale.x = 0.02
        boundary_marker.color = ColorRGBA(1.0, 1.0, 1.0, 0.3)  # White, very transparent
        
        # Create box edges
        bounds = [(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0),
                  (-10, -10, 10), (10, -10, 10), (10, 10, 10), (-10, 10, 10)]
        
        edges = [(0,1), (1,2), (2,3), (3,0),  # Bottom
                 (4,5), (5,6), (6,7), (7,4),  # Top
                 (0,4), (1,5), (2,6), (3,7)]  # Sides
        
        for edge in edges:
            p1 = Point(x=bounds[edge[0]][0], y=bounds[edge[0]][1], z=bounds[edge[0]][2])
            p2 = Point(x=bounds[edge[1]][0], y=bounds[edge[1]][1], z=bounds[edge[1]][2])
            boundary_marker.points.append(p1)
            boundary_marker.points.append(p2)
        
        marker_array.markers.append(boundary_marker)
        
        # Publish all markers
        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        visualizer = PathVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass