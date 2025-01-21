import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from visualization_setup import update_train_agent

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc_input_dim = 64 * 8 * 8
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class WeatherSimulator:
    def __init__(self):
        self.weather_conditions = ['clear', 'rain', 'snow']
        self.current_weather = 'clear'
        self.weather_duration = 0
        self.weather_change_probability = 0.01
        self.current_friction = 1.0
        self.friction_change_rate = random.uniform(-0.001, 0.001)
        self.friction_ranges = {
            'clear': (0.95, 1.0),
            'rain': (0.6, 0.8),
            'snow': (0.4, 0.6)
        }

    def update(self):
        if random.random() < self.weather_change_probability:
            self.current_weather = random.choice(self.weather_conditions)
            self.weather_duration = random.randint(100, 300)
            min_friction, max_friction = self.friction_ranges[self.current_weather]
            self.current_friction = random.uniform(min_friction, max_friction)
        else:
            self.weather_duration = max(0, self.weather_duration - 1)
            new_friction = self.current_friction + self.friction_change_rate
            min_friction, max_friction = self.friction_ranges[self.current_weather]
            self.current_friction = np.clip(new_friction, min_friction, max_friction)

    def get_friction_factor(self):
        return self.current_friction

    def get_current_conditions(self):
        return {
            'weather': self.current_weather,
            'friction': self.current_friction,
            'duration': self.weather_duration
        }

class EnhancedCarRacingEnv:
    def __init__(self, env):
        self.env = env
        self.weather = WeatherSimulator()
        self.max_fuel = 1000.0
        self.fuel = self.max_fuel
        self.prev_speed = 0
        self.speed_history = deque(maxlen=10)
        self.discrete_actions = {
            0: [0.0, 0.0, 0.0],    # No action
            1: [-1.0, 0.0, 0.0],   # Left
            2: [1.0, 0.0, 0.0],    # Right
            3: [0.0, 1.0, 0.0],    # Gas
            4: [0.0, 0.0, 0.8]     # Brake
        }
        # Initialize reward components
        self.last_original_reward = 0
        self.last_fuel_penalty = 0
        self.last_weather_penalty = 0
        self.last_speed_consistency_reward = 0
        self.last_acceleration_penalty = 0

    def reset(self):
        state, info = self.env.reset()
        self.fuel = self.max_fuel
        self.prev_speed = 0
        self.speed_history.clear()
        self.weather = WeatherSimulator()
        return state, info

    def step(self, discrete_action):
        continuous_action = np.array(self.discrete_actions[discrete_action], dtype=np.float32)
        next_state, reward, done, truncated, info = self.env.step(continuous_action)
        speed = np.mean(next_state[84:94, 12, 0]) / 255.0 * 100
        enhanced_reward = self.calculate_reward(reward, discrete_action, speed)
        
        if self.fuel <= 0:
            done = True

        info.update({
            'fuel_level': self.fuel,
            'weather': self.weather.current_weather,
            'friction': self.weather.get_friction_factor()
        })

        return next_state, enhanced_reward, done, truncated, info

    def calculate_reward(self, original_reward, action, speed):
        self.weather.update()
        friction_factor = self.weather.get_friction_factor()

        # Store original reward
        self.last_original_reward = original_reward * friction_factor

        # Calculate fuel penalty
        fuel_consumption = 0.5 if action == 3 else 0.1
        self.fuel -= fuel_consumption
        self.last_fuel_penalty = -2.0 if self.fuel <= 0 else 0

        # Calculate speed consistency reward
        self.speed_history.append(speed)
        speed_variance = np.var(list(self.speed_history)) if len(self.speed_history) > 1 else 0
        self.last_speed_consistency_reward = -0.1 * speed_variance

        # Calculate weather penalty
        if self.weather.current_weather != 'clear':
            self.last_weather_penalty = -0.2 * max(0, speed - 50) if speed > 50 else 0
        else:
            self.last_weather_penalty = 0

        # Calculate acceleration penalty
        acceleration = speed - self.prev_speed
        self.last_acceleration_penalty = -0.1 * abs(acceleration)
        self.prev_speed = speed

        total_reward = (
            self.last_original_reward +
            self.last_fuel_penalty +
            self.last_speed_consistency_reward +
            self.last_weather_penalty +
            self.last_acceleration_penalty
        )

        return total_reward

    def get_reward_components(self):
        return {
            'original': self.last_original_reward,
            'fuel': self.last_fuel_penalty,
            'weather': self.last_weather_penalty,
            'speed': self.last_speed_consistency_reward,
            'acceleration': self.last_acceleration_penalty
        }

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.policy_net = DQN(action_dim).to(self.device)
        self.target_net = DQN(action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state = self.preprocess_state(state)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def preprocess_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)
        return state.to(self.device)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def get_average_q_value(self):
        if len(self.memory) < self.batch_size:
            return 0
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([s[0] for s in batch])).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(states)
            return q_values.mean().item()

if __name__ == "__main__":
    # Update the train_agent function with visualization support
    train_agent = update_train_agent()
    # Run training
    train_agent()
