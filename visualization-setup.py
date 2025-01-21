import os
import json
import torch
import gymnasium as gym
import numpy as np
from pathlib import Path

def save_metrics(metrics, save_dir='training_metrics'):
    """Save metrics to a JSON file"""
    os.makedirs(save_dir, exist_ok=True)
    metrics_file = os.path.join(save_dir, 'training_metrics.json')
    
    # Convert numpy/torch values to Python native types
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (list, dict)):
            serializable_metrics[key] = value
        else:
            serializable_metrics[key] = float(value)
    
    with open(metrics_file, 'w') as f:
        json.dump(serializable_metrics, f)

def collect_weight_stats(model):
    """Collect statistics about model weights"""
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'max': param.data.max().item(),
                'min': param.data.min().item()
            }
    return stats

def update_train_agent():
    def train_agent():
        base_env = gym.make("CarRacing-v3")
        env = EnhancedCarRacingEnv(base_env)
        agent = DQNAgent(state_dim=(96, 96, 3), action_dim=5)
        
        # Create metrics directory
        metrics_dir = Path('training_metrics')
        metrics_dir.mkdir(exist_ok=True)
        
        # Initialize metrics storage
        training_metrics = {
            'episodes': [],
            'weights': [],
            'rewards': {
                'total': [],
                'components': []
            },
            'parameters': [],
            'action_distribution': [0] * 5
        }
        
        num_episodes = 1000
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_rewards_components = {
                'original': 0,
                'fuel': 0,
                'weather': 0,
                'speed': 0,
                'acceleration': 0
            }
            episode_actions = [0] * 5
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Track action distribution
                episode_actions[action] += 1
                
                # Track reward components
                reward_components = env.get_reward_components()
                for key in episode_rewards_components:
                    episode_rewards_components[key] += reward_components[key]

                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.update()

                state = next_state
                episode_reward += reward

                if done or truncated:
                    break

            # Collect and store metrics
            training_metrics['episodes'].append(episode)
            training_metrics['weights'].append(collect_weight_stats(agent.policy_net))
            training_metrics['rewards']['total'].append(episode_reward)
            training_metrics['rewards']['components'].append(episode_rewards_components)
            training_metrics['parameters'].append({
                'epsilon': agent.epsilon,
                'loss': loss if loss is not None else 0,
                'avg_q_value': agent.get_average_q_value()
            })
            
            # Normalize action distribution
            total_actions = sum(episode_actions)
            training_metrics['action_distribution'].append(
                [count/total_actions for count in episode_actions]
            )

            # Save metrics every 10 episodes
            if episode % 10 == 0:
                save_metrics(training_metrics)
                print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

            # Save model checkpoint every 100 episodes
            if (episode + 1) % 100 == 0:
                checkpoint_path = metrics_dir / f'model_checkpoint_{episode+1}.pth'
                torch.save({
                    'model_state_dict': agent.policy_net.state_dict(),
                    'metrics': training_metrics
                }, checkpoint_path)

        env.close()
        return training_metrics

    return train_agent
