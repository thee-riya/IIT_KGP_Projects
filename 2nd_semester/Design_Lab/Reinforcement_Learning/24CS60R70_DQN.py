import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import time
from matplotlib import colors
from tqdm import tqdm
import joblib
import gym
from gym import spaces
import urllib.request
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

# Twemoji CDN URLs
EMOJI_URLS = {
    "agent": "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f9d1.png",
    "dirt": "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f4a9.png",
    "obstacle": "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f6a7.png",
}
EMOJI_CACHE = {}

def get_emoji_image(role):
    if role in EMOJI_CACHE:
        return EMOJI_CACHE[role]
    url = EMOJI_URLS[role]
    local_path = f"{role}.png"
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
    img = mpimg.imread(local_path)
    EMOJI_CACHE[role] = img
    return img

def generate_obstacles(grid_size, density=0.1):
    obstacles = []
    for x in range(1, grid_size - 1):
        for y in range(1, grid_size - 1):
            if np.random.rand() < density:
                obstacles.append((x, y))
    return obstacles

class CleaningEnv(gym.Env):
    """
    Custom Gym environment for a room cleaning task.
    
    Grid cells:
      - Empty: white.
      - Dirt: 💩 placed at a random edge.
      - Obstacles: 🚧 (generated randomly each reset if not provided).
      - Agent: 🧑.
      - Walls: drawn as thick black borders.
    
    Reward Structure:
      - +10 for reaching the dirt.
      - -5 for hitting an obstacle or invalid move.
      - -1 for every valid move.
    
    Action Space:
       0: Up, 1: Down, 2: Left, 3: Right.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=10, obstacles=None, obstacle_density=0.1):
        super(CleaningEnv, self).__init__()
        self.grid_size = grid_size
        self.fixed_obstacles = obstacles
        self.obstacle_density = obstacle_density
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)
        self.agent_pos = np.array([0, 0])
        self.dirt_pos = self._place_dirt_near_wall()

        self.fig, self.ax = plt.subplots()
        self.cmap = colors.ListedColormap(['white', 'gray'])
        self.norm = colors.BoundaryNorm([0, 1, 2], self.cmap.N)
        plt.ion()

    def _place_dirt_near_wall(self):
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            return (np.random.randint(0, self.grid_size), 0)
        elif edge == 'bottom':
            return (np.random.randint(0, self.grid_size), self.grid_size - 1)
        elif edge == 'left':
            return (0, np.random.randint(0, self.grid_size))
        else:
            return (self.grid_size - 1, np.random.randint(0, self.grid_size))

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.dirt_pos = self._place_dirt_near_wall()
        if self.fixed_obstacles is None:
            self.obstacles = set(generate_obstacles(self.grid_size, density=self.obstacle_density))
        else:
            self.obstacles = set(self.fixed_obstacles)
        return self.agent_pos.copy()

    def step(self, action):
        new_pos = self.agent_pos.copy()
        if action == 0 and self.agent_pos[1] > 0:
            new_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:
            new_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:
            new_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:
            new_pos[0] += 1

        reward = -1
        done = False
        if tuple(new_pos) in self.obstacles:
            reward = -5
            new_pos = self.agent_pos.copy()
        else:
            self.agent_pos = new_pos

        if tuple(self.agent_pos) == self.dirt_pos:
            reward = 10
            done = True

        return self.agent_pos.copy(), reward, done, {}

    def render(self, mode="human", action=None, reward=None, episode=None):
        if self.grid_size > 50:
            return
        self.ax.clear()
        grid_bg = np.zeros((self.grid_size, self.grid_size))
        self.ax.imshow(grid_bg, cmap=self.cmap, norm=self.norm)

        def draw_emoji(img, pos):
            self.ax.imshow(img, extent=[pos[0] - 0.5, pos[0] + 0.5,
                                         pos[1] - 0.5, pos[1] + 0.5], zorder=5)

        draw_emoji(get_emoji_image("agent"), self.agent_pos)
        draw_emoji(get_emoji_image("dirt"), self.dirt_pos)
        for (x, y) in self.obstacles:
            draw_emoji(get_emoji_image("obstacle"), (x, y))

        for x in range(self.grid_size + 1):
            self.ax.plot([x - 0.5, x - 0.5], [-0.5, self.grid_size - 0.5], color="black", linewidth=2)
        for y in range(self.grid_size + 1):
            self.ax.plot([-0.5, self.grid_size - 0.5], [y - 0.5, y - 0.5], color="black", linewidth=2)

        title = f"Grid: {self.grid_size}x{self.grid_size}"
        if episode is not None:
            title += f" | Episode: {episode}"
        if action is not None:
            action_names = ['Up', 'Down', 'Left', 'Right']
            if 0 <= action < len(action_names):
                title += f" | Action: {action_names[action]}"
        if reward is not None:
            title += f" | Reward: {reward}"
        self.ax.set_title(title)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.pause(0.001)

# Simple neural network approximator for DQN.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Replay Buffer for experience replay.
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env, learning_rate=1e-3, gamma=0.99, epsilon=0.1, buffer_size=1e5,
                 batch_size=64, target_update_freq=1000, max_steps=1000):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_steps = max_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 2  # (x, y) state
        self.output_dim = env.action_space.n

        self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        current_q = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            max_next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q = reward_batch + (self.gamma * max_next_q * (1 - done_batch))
        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=1000, render_interval=50):
        episode_rewards = []
        for ep in tqdm(range(episodes), desc="Training DQN"):
            state = self.env.reset()
            total_reward = 0
            for t in range(self.max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.optimize_model()
                self.steps_done += 1
                if self.steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if render_interval and (ep % render_interval == 0):
                    self.env.render(action, reward, episode=ep)
                if done:
                    break
            episode_rewards.append(total_reward)
            self.epsilon = max(0.01, self.epsilon * 0.995)
        return episode_rewards

def simple_hyperparameter_search_dqn(env):
    print("Starting quick hyperparameter search for DQN...")
    candidate_lrs = [1e-3, 5e-4]
    candidate_gammas =[0.9, 0.95, 0.99]
    candidate_epsilons = [0.05, 0.1, 0.2, 0.4, 0.5]
    candidate_batch_sizes = [32, 64]
    candidate_target_update = [1000, 2000]
    candidate_max_steps = [100, 200]
    best_avg = -np.inf
    best_params = None
    search_episodes = 20

    for lr in candidate_lrs:
        for gamma in candidate_gammas:
            for eps in candidate_epsilons:
                for batch in candidate_batch_sizes:
                    for tuf in candidate_target_update:
                        for ms in candidate_max_steps:
                            agent = DQNAgent(env, learning_rate=lr, gamma=gamma, epsilon=eps,
                                               batch_size=batch, target_update_freq=tuf,
                                               max_steps=ms)
                            rewards = agent.train(episodes=search_episodes, render_interval=search_episodes+1)
                            avg_reward = np.mean(rewards)
                            print(f"lr={lr}, gamma={gamma}, ε={eps}, batch={batch}, tuf={tuf}, ms={ms} -> Avg Reward: {avg_reward:.2f}")
                            if avg_reward > best_avg:
                                best_avg = avg_reward
                                best_params = {'learning_rate': lr, 'gamma': gamma, 'epsilon': eps,
                                               'batch_size': batch, 'target_update_freq': tuf, 'max_steps': ms}
    print("Best hyperparameters:", best_params)
    return best_params

def evaluation_mode_dqn():
    grid_sizes = [10, 100, 1000, 10000]
    eval_results = {}
    for size in grid_sizes:
        print(f"\nEvaluating grid size: {size}x{size}")
        obstacles = set(generate_obstacles(size, density=0.1))
        env = CleaningEnv(grid_size=size, obstacles=obstacles)
        agent = DQNAgent(env)
        start_time = time.time()
        rewards = agent.train(episodes=10)
        elapsed = time.time() - start_time
        if size <= 50:
            plt.pause(2)
            plt.close(env.fig)
        eval_results[size] = elapsed
        print(f"Time taken for grid size {size}: {elapsed:.2f} seconds")
    return eval_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', type=int, default=10, help="Size of the grid")
    parser.add_argument('--episodes', type=int, default=200, help="Number of training episodes")
    parser.add_argument('--hyperparam', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--eval', action='store_true', help='Run evaluation mode over multiple grid sizes')
    parser.add_argument('--save', type=str, help='Path to save trained model')
    args = parser.parse_args()

    if args.eval:
        eval_results = evaluation_mode_dqn()
        print("Evaluation Results (Grid Size : Time in sec):")
        for size, t in eval_results.items():
            print(f"{size} : {t:.2f} sec")
    else:
        env = CleaningEnv(grid_size=args.grid_size, obstacles=None, obstacle_density=0.1)
        if args.hyperparam:
            best_params = simple_hyperparameter_search_dqn(env)
            agent = DQNAgent(env, learning_rate=best_params['learning_rate'], gamma=best_params['gamma'],
                              epsilon=best_params['epsilon'], batch_size=best_params['batch_size'],
                              target_update_freq=best_params['target_update_freq'], max_steps=best_params['max_steps'])
        else:
            agent = DQNAgent(env)
        rewards = agent.train(episodes=args.episodes)
        if args.save:
            joblib.dump(agent, args.save)
            print("DQN model saved at", args.save)
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Progress")
        plt.show(block=True)

if __name__ == "__main__":
    main()
