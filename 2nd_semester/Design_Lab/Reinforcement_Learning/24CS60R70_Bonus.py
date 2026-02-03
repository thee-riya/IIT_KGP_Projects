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

# Twemoji CDN URLs
EMOJI_URLS = {
    "agent": "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f9d1.png",
    "dirt": "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f4a9.png",
    "obstacle": "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f6a7.png",
}
EMOJI_CACHE = {}

def get_emoji_image(role):
    """Downloads and returns emoji image from Twemoji CDN."""
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
    """
    Generate obstacles randomly in interior cells (not on the border).
    """
    obstacles = []
    for x in range(1, grid_size - 1):
        for y in range(1, grid_size - 1):
            if np.random.rand() < density:
                obstacles.append((x, y))
    return obstacles

def _place_dirt_near_wall(grid_size):
    """
    Place a dirt cell on a random edge of the grid.
    """
    edge = np.random.choice(['top', 'bottom', 'left', 'right'])
    if edge == 'top':
        return (np.random.randint(0, grid_size), 0)
    elif edge == 'bottom':
        return (np.random.randint(0, grid_size), grid_size - 1)
    elif edge == 'left':
        return (0, np.random.randint(0, grid_size))
    else:
        return (grid_size - 1, np.random.randint(0, grid_size))

class CleaningEnv(gym.Env):
    """
    Custom Gym environment for a room cleaning task with multiple dirt cells.
    
    Grid cells:
      - Empty: white.
      - Dirt: 💩 placed on one or more edges.
      - Obstacles: 🚧 (generated randomly each reset if not provided).
      - Agent: 🧑.
      - Walls: drawn as thick black borders.
    
    Reward Structure:
      - +100 for cleaning a dirt cell.
      - -5 for hitting an obstacle or invalid move.
      - -1 for every valid move.
    
    Episode terminates when all dirt cells are cleaned.
    
    Action Space:
       0: Up, 1: Down, 2: Left, 3: Right.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size=10, num_dirts=1, obstacles=None, obstacle_density=0.1):
        super(CleaningEnv, self).__init__()
        self.grid_size = grid_size
        self.num_dirts = num_dirts
        self.fixed_obstacles = obstacles
        self.obstacle_density = obstacle_density
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)
        self.agent_pos = np.array([0, 0])
        self.dirt_positions = []  # will be set in reset()

        self.fig, self.ax = plt.subplots()
        self.cmap = colors.ListedColormap(['white', 'gray'])
        self.norm = colors.BoundaryNorm([0, 1, 2], self.cmap.N)
        plt.ion()

    def reset(self):
        self.agent_pos = np.array([0, 0])
        # Generate the specified number of dirt cells (ensuring they are unique)
        self.dirt_positions = []
        while len(self.dirt_positions) < self.num_dirts:
            pos = _place_dirt_near_wall(self.grid_size)
            if pos not in self.dirt_positions:
                self.dirt_positions.append(pos)
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

        # Check if the agent cleans any dirt cell
        if tuple(self.agent_pos) in self.dirt_positions:
            self.dirt_positions.remove(tuple(self.agent_pos))
            reward = 100
            # Episode done only when all dirt cells are cleaned.
            if not self.dirt_positions:
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
        # Draw all dirt cells
        for dirt in self.dirt_positions:
            draw_emoji(get_emoji_image("dirt"), dirt)
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
        title += f" | Dirt remaining: {len(self.dirt_positions)}"
        self.ax.set_title(title)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.pause(0.001)

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: key: (state, action)
        # NOTE: With multiple dirt cells the full state ideally would include their positions.
        # Here we use a simplified state representation based solely on the agent's position.
        self.q_table = {}

    def train(self, episodes=1000, max_steps=100, render_interval=50):
        rewards = []
        for ep in tqdm(range(episodes), desc="Training Q-Learning"):
            state = tuple(self.env.reset())
            total_reward = 0
            for step in range(max_steps):
                action = self._choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if ep % render_interval == 0:
                    self.env.render(action, reward, ep)
                self._update_q_table(state, action, reward, tuple(next_state))
                total_reward += reward
                state = tuple(next_state)
                if done:
                    break
            rewards.append(total_reward)
            self.epsilon = max(0.01, self.epsilon * 0.995)
        return rewards

    def _choose_action(self, state):
        # Heuristic: If there is at least one dirt, choose the one nearest to the agent.
        x, y = state
        if self.env.dirt_positions:
            # Find the nearest dirt using Manhattan distance.
            distances = [abs(x - dx) + abs(y - dy) for (dx, dy) in self.env.dirt_positions]
            target = self.env.dirt_positions[np.argmin(distances)]
            dx, dy = target[0] - x, target[1] - y
            # Prefer horizontal movement if difference is larger.
            if abs(dx) > abs(dy):
                desired = 3 if dx > 0 else 2
            elif dy != 0:
                desired = 1 if dy > 0 else 0
            else:
                desired = None

            if desired is not None:
                new_state = list(state)
                if desired == 0 and y > 0:
                    new_state[1] = y - 1
                elif desired == 1 and y < self.env.grid_size - 1:
                    new_state[1] = y + 1
                elif desired == 2 and x > 0:
                    new_state[0] = x - 1
                elif desired == 3 and x < self.env.grid_size - 1:
                    new_state[0] = x + 1
                if tuple(new_state) not in self.env.obstacles:
                    return desired

        # If heuristic is not applied or blocked, use standard ε-greedy.
        if np.random.random() < self.epsilon:
            return np.random.randint(4)
        q_values = [self.q_table.get((state, a), 0) for a in range(4)]
        return int(np.argmax(q_values))

    def _update_q_table(self, state, action, reward, next_state):
        old_q = self.q_table.get((state, action), 0)
        max_next_q = max([self.q_table.get((next_state, a), 0) for a in range(4)])
        self.q_table[(state, action)] = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

def simple_hyperparameter_search(env):
    print("Starting quick hyperparameter search for Q-Learning...")
    candidate_alphas = [0.05, 0.1,0.2]
    candidate_gammas = [0.9, 0.95, 0.99]
    candidate_epsilons = [0.05, 0.1, 0.2, 0.4, 0.5]
    candidate_max_steps = [10, 100, 1000]
    best_avg = -np.inf
    best_params = None
    search_episodes = 20  # few episodes for fast tuning

    for a in candidate_alphas:
        for g in candidate_gammas:
            for e in candidate_epsilons:
                for ms in candidate_max_steps:
                    agent = QLearningAgent(env, alpha=a, gamma=g, epsilon=e)
                    rewards = agent.train(episodes=search_episodes, max_steps=ms, render_interval=search_episodes+1)
                    avg_reward = np.mean(rewards)
                    print(f"α={a}, γ={g}, ε={e}, max_steps={ms} -> Avg Reward: {avg_reward:.2f}")
                    if avg_reward > best_avg:
                        best_avg = avg_reward
                        best_params = {'alpha': a, 'gamma': g, 'epsilon': e, 'max_steps': ms}
    print("Best hyperparameters:", best_params)
    return best_params

def evaluation_mode_qlearning():
    grid_sizes = [10, 100, 1000, 10000]
    eval_results = {}
    for size in grid_sizes:
        print(f"\nEvaluating grid size: {size}x{size}")
        obstacles = set(generate_obstacles(size, density=0.1))
        env = CleaningEnv(grid_size=size, num_dirts=1, obstacles=obstacles)
        agent = QLearningAgent(env)
        start_time = time.time()
        rewards = agent.train(episodes=10, max_steps=100)
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
    parser.add_argument('--num_dirts', type=int, default=1, help="Number of dirt cells to generate")
    parser.add_argument('--hyperparam', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--eval', action='store_true', help='Run evaluation mode over multiple grid sizes')
    args = parser.parse_args()

    if args.eval:
        eval_results = evaluation_mode_qlearning()
        print("Evaluation Results (Grid Size : Time in sec):")
        for size, t in eval_results.items():
            print(f"{size} : {t:.2f} sec")
    else:
        env = CleaningEnv(grid_size=args.grid_size, num_dirts=args.num_dirts, obstacles=None, obstacle_density=0.1)
        if args.hyperparam:
            best_params = simple_hyperparameter_search(env)
            agent = QLearningAgent(env, alpha=best_params['alpha'],
                                   gamma=best_params['gamma'],
                                   epsilon=best_params['epsilon'])
            rewards = agent.train(episodes=args.episodes, max_steps=best_params['max_steps'])
        else:
            agent = QLearningAgent(env)
            rewards = agent.train(episodes=args.episodes)
        plt.figure()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Q-Learning Training Progress")
        plt.show(block=True)
        # Optionally, save the model or Q-table:
        # joblib.dump(agent, "q_learning_model.pkl")
        # save_q_table(agent, filename="q_table.png")

if __name__ == "__main__":
    main()
