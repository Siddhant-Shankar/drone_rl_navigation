import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
import time
import numpy as np

class NavigationTester:
    """Test the drone navigation system"""
    
    def __init__(self):
        rospy.init_node('navigation_tester', anonymous=True)
        
        # Publishers
        self.goal_pub = rospy.Publisher('/drone/goal', Point, queue_size=10)
        
        # Subscribers
        self.position_sub = rospy.Subscriber('/drone/position', Point, self.position_callback)
        self.reward_sub = rospy.Subscriber('/drone/reward', Float64, self.reward_callback)
        
        # State
        self.current_position = None
        self.current_reward = 0
        self.test_goals = [
            Point(5.0, 5.0, 3.0),
            Point(-5.0, 5.0, 4.0),
            Point(-5.0, -5.0, 2.0),
            Point(5.0, -5.0, 5.0),
            Point(0.0, 0.0, 3.0)
        ]
        
        rospy.sleep(2)  # Wait for initialization
        rospy.loginfo("Navigation Tester initialized")
    
    def position_callback(self, msg):
        """Track current position"""
        self.current_position = msg
    
    def reward_callback(self, msg):
        """Track reward"""
        self.current_reward = msg.data
    
    def send_goal(self, goal):
        """Send goal to controller"""
        rospy.loginfo(f"Sending goal: [{goal.x:.2f}, {goal.y:.2f}, {goal.z:.2f}]")
        self.goal_pub.publish(goal)
    
    def wait_for_goal(self, goal, timeout=60, threshold=1.0):
        """Wait until goal is reached or timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.current_position:
                dist = np.sqrt(
                    (self.current_position.x - goal.x)**2 +
                    (self.current_position.y - goal.y)**2 +
                    (self.current_position.z - goal.z)**2
                )
                
                if dist < threshold:
                    elapsed = time.time() - start_time
                    rospy.loginfo(f"✓ Goal reached in {elapsed:.2f}s! Distance: {dist:.2f}m")
                    return True
                
                # Log progress
                if int(time.time() - start_time) % 5 == 0:
                    rospy.loginfo(f"Distance to goal: {dist:.2f}m, Reward: {self.current_reward:.2f}")
            
            rospy.sleep(0.1)
        
        rospy.logwarn(f"✗ Goal not reached within {timeout}s timeout")
        return False
    
    def run_test_sequence(self):
        """Run through test goal sequence"""
        rospy.loginfo("="*50)
        rospy.loginfo("Starting Navigation Test Sequence")
        rospy.loginfo("="*50)
        
        results = []
        
        for i, goal in enumerate(self.test_goals):
            rospy.loginfo(f"\nTest {i+1}/{len(self.test_goals)}")
            self.send_goal(goal)
            
            success = self.wait_for_goal(goal, timeout=60)
            results.append(success)
            
            rospy.sleep(2)  # Brief pause between goals
        
        # Print summary
        rospy.loginfo("\n" + "="*50)
        rospy.loginfo("Test Summary")
        rospy.loginfo("="*50)
        success_count = sum(results)
        rospy.loginfo(f"Goals reached: {success_count}/{len(results)}")
        rospy.loginfo(f"Success rate: {success_count/len(results)*100:.1f}%")
        rospy.loginfo("="*50)
    
    def run_random_goals(self, num_goals=10):
        """Test with random goals"""
        rospy.loginfo("="*50)
        rospy.loginfo(f"Random Goals Test ({num_goals} goals)")
        rospy.loginfo("="*50)
        
        results = []
        
        for i in range(num_goals):
            # Generate random goal within bounds
            goal = Point(
                x=np.random.uniform(-8, 8),
                y=np.random.uniform(-8, 8),
                z=np.random.uniform(1, 8)
            )
            
            rospy.loginfo(f"\nRandom Test {i+1}/{num_goals}")
            self.send_goal(goal)
            
            success = self.wait_for_goal(goal, timeout=60)
            results.append(success)
            
            rospy.sleep(2)
        
        # Print summary
        rospy.loginfo("\n" + "="*50)
        rospy.loginfo("Random Test Summary")
        rospy.loginfo("="*50)
        success_count = sum(results)
        rospy.loginfo(f"Goals reached: {success_count}/{len(results)}")
        rospy.loginfo(f"Success rate: {success_count/len(results)*100:.1f}%")
        rospy.loginfo("="*50)

def main():
    """Main function"""
    tester = NavigationTester()
    
    # Run test sequence
    try:
        tester.run_test_sequence()
        rospy.sleep(5)
        tester.run_random_goals(num_goals=5)
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Test interrupted by user")

if __name__ == '__main__':
    main()