import rospy
import numpy as np
from std_msgs.msg import Float64, String
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from drone_rl_navigation.environment import DroneEnvironment
from drone_rl_navigation.models import TD3Agent

class TrainingNode:
    """ROS node for training the RL agent"""
    
    def __init__(self):
        rospy.init_node('training_node', anonymous=True)
        
        # Parameters
        self.num_episodes = rospy.get_param('~num_episodes', 500)
        self.max_steps = rospy.get_param('~max_steps', 1000)
        self.save_frequency = rospy.get_param('~save_frequency', 50)
        self.model_dir = rospy.get_param('~model_dir', 'models/')
        
        # Create model directory
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Initialize environment and agent
        self.env = DroneEnvironment(max_steps=self.max_steps)
        self.agent = TD3Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        # Publishers
        self.reward_pub = rospy.Publisher('/training/episode_reward', Float64, queue_size=10)
        self.loss_pub = rospy.Publisher('/training/loss', String, queue_size=10)
        self.progress_pub = rospy.Publisher('/training/progress', String, queue_size=10)
        
        # Training statistics
        self.episode_rewards = []
        self.success_count = 0
        
        rospy.loginfo("Training Node initialized")
        rospy.loginfo(f"Training for {self.num_episodes} episodes")
    
    def train_episode(self, episode):
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < self.max_steps:
            # Select action with exploration noise
            noise = max(0.1, 0.5 - episode / (self.num_episodes * 0.5))  # Decay noise
            action = self.agent.select_action(state, noise=noise)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            
            # Store transition
            self.agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train agent
            if len(self.agent.replay_buffer) > self.agent.batch_size:
                actor_loss, critic_loss = self.agent.train()
            
            state = next_state
            step += 1
            
            # Check success
            if done and reward > 50:  # Goal reached
                self.success_count += 1
        
        return episode_reward, step, info
    
    def train(self):
        """Main training loop"""
        rospy.loginfo("Starting training...")
        
        for episode in range(self.num_episodes):
            if rospy.is_shutdown():
                break
            
            # Train one episode
            episode_reward, steps, info = self.train_episode(episode)
            self.episode_rewards.append(episode_reward)
            
            # Publish reward
            self.reward_pub.publish(Float64(episode_reward))
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                success_rate = self.success_count / (episode + 1) * 100
                
                progress_msg = json.dumps({
                    'episode': episode,
                    'reward': float(episode_reward),
                    'avg_reward': float(avg_reward),
                    'success_rate': float(success_rate),
                    'steps': steps,
                    'goal_distance': float(info['goal_distance'])
                })
                self.progress_pub.publish(String(progress_msg))
                
                rospy.loginfo(f"Episode {episode}/{self.num_episodes}: "
                            f"Reward={episode_reward:.2f}, "
                            f"Avg={avg_reward:.2f}, "
                            f"Success={success_rate:.1f}%, "
                            f"Steps={steps}")
            
            # Publish losses
            if len(self.agent.actor_loss_history) > 0 and episode % 5 == 0:
                loss_msg = json.dumps({
                    'actor_loss': float(self.agent.actor_loss_history[-1]) if self.agent.actor_loss_history else 0,
                    'critic_loss': float(self.agent.critic_loss_history[-1]) if self.agent.critic_loss_history else 0
                })
                self.loss_pub.publish(String(loss_msg))
            
            # Save model periodically
            if (episode + 1) % self.save_frequency == 0:
                model_path = os.path.join(self.model_dir, f'agent_episode_{episode+1}.npy')
                self.agent.save_model(model_path)
                rospy.loginfo(f"Model saved at episode {episode+1}")
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, 'trained_agent.npy')
        self.agent.save_model(final_model_path)
        
        # Print final statistics
        avg_final_reward = np.mean(self.episode_rewards[-50:])
        final_success_rate = self.success_count / self.num_episodes * 100
        
        rospy.loginfo("="*50)
        rospy.loginfo("Training Complete!")
        rospy.loginfo(f"Total Episodes: {self.num_episodes}")
        rospy.loginfo(f"Average Reward (last 50): {avg_final_reward:.2f}")
        rospy.loginfo(f"Success Rate: {final_success_rate:.1f}%")
        rospy.loginfo(f"Final model saved to: {final_model_path}")
        rospy.loginfo("="*50)
        
        # Save training statistics
        stats_path = os.path.join(self.model_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump({
                'episode_rewards': self.episode_rewards,
                'num_episodes': self.num_episodes,
                'success_count': self.success_count,
                'success_rate': final_success_rate,
                'avg_reward': float(avg_final_reward)
            }, f, indent=2)
        rospy.loginfo(f"Training statistics saved to: {stats_path}")

if __name__ == '__main__':
    try:
        trainer = TrainingNode()
        trainer.train()
    except rospy.ROSInterruptException:
        pass