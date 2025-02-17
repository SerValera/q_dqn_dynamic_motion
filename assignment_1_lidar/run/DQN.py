import numpy as np
import random
from env import *
from lidar import *

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import imageio

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

GAMMA = 0.99
LR = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        
    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DNQSolver:
    def __init__(self):
        # Env setup
        self.grid_size = (10, 20)
        self.init_state = [3, 3, 0]
        self.goal_state = [9, 18]
        self.state = self.init_state

        # Obst setup
        self.obstacales_list = []
        self.env = GridEnvironment(self.grid_size, self.init_state, self.goal_state, self.obstacales_list)
        num_obstacles = 20
        self.obstacales_list = self.env.generate_random_obstacles(num_obstacles)

        # True - Dynamic env
        # False - Static env (deafult)
        self.env.is_move_obst = True

        # to create gif
        self.is_create_gif = True
        self.is_show_each_step_agent = False #doesn't work well, keep false
        self.env.is_show_all_history_states = True

        # Action list
        self.actions = [[0, 1, 0], [-1, 1, 0], [-1, 0, 0], [-1, -1, 0],    #Forward, Back, Right, Left
                        [0, -1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0]   #FR, FL, BR, BL 
                        ] 

        # To analyse learning
        self.learning_history = []

        # Init lidar
        self.lidar = LidarObservation()
        self.lidar.update_lidar_state(self.init_state, self.obstacales_list, self.grid_size, self.goal_state)
        self.env.state_history.append(self.init_state)
        self.env.plot_state_history()
        self.lidar.show_lidar_grid()

        self.agent = DQNAgent(9, len(self.actions))

        self.state_env_prev = self.init_state 
        self.state_env_new = [0,0,0]
        self.inter_in_episode = 1000
        self.cpinter_episodes = 0

        self.steps_history = []

    def plot_learning_history(self, history, filename="reward_history.png"):
        epochs, values = zip(*history)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values, marker='o', markersize=1, linestyle='-', color='b', label='Learning Progress')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Learning History')
        plt.legend()
        plt.grid(True)

        save_dir = os.path.join(os.path.dirname(__file__), "..", "analysis")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"saved to {save_path}")

        plt.close()

    def plot_steps_history(self, history, filename="steps_history.png"):
        epochs, values = zip(*history)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values, marker='o', markersize=1, linestyle='-', color='b', label='Learning Progress')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Learning Steps History')
        plt.legend()
        plt.grid(True)

        save_dir = os.path.join(os.path.dirname(__file__), "..", "analysis")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"saved to {save_path}")

        plt.close()


    def learnong_dqn(self):
        num_episodes = 650
        for episode in range(num_episodes):
            self.lidar.update_lidar_state(self.init_state, self.obstacales_list, self.grid_size, self.goal_state)
            self.state = self.lidar.state_lidar
            self.state_env_prev = self.init_state 
            self.state_env_new = [0,0,0]

            done = False
            total_reward = 0
            epsilon = max(EPSILON * (EPSILON_DECAY ** episode), EPSILON_MIN)
            # epsilon = 0.9

            self.inter_in_episode = 250
            self.cpinter_episodes = 0

            while not done:
                action = self.agent.select_action(self.state, epsilon)
                self.state_env_new, r, done = self.env.update_state(self.state_env_prev, self.actions[action])
                self.lidar.update_lidar_state(self.state_env_new, self.env.obstacales, self.grid_size, self.goal_state)

                next_state = self.lidar.state_lidar

                self.agent.store_experience(self.state, action, r, next_state, done)
                self.agent.train()
                
                self.state = next_state
                total_reward += r

                self.state_env_prev = self.state_env_new

                

                self.cpinter_episodes += 1
                self.steps_history.append([episode, self.cpinter_episodes])

                if self.cpinter_episodes == self.inter_in_episode:
                    done = True
            
            self.learning_history.append([episode, total_reward])

            if episode % TARGET_UPDATE == 0:
                self.agent.update_target_network()
                
            print(f"Episod: {episode}, Reward: {total_reward}, Epsilon: {epsilon}")

        self.plot_learning_history(self.learning_history)
        self.plot_steps_history(self.steps_history)

    def run_agent_dqn(self):
        self.lidar.update_lidar_state(self.init_state, self.obstacales_list, self.grid_size, self.goal_state)
        self.state = self.lidar.state_lidar
    
        self.state_env_prev = self.init_state 
        self.state_env_new = [0,0,0]

        print("Agent run")
        frame_id = 0
        reward = 0
        epsilon = 0

        while self.state_env_new[:2] != self.goal_state:
            action_id = self.agent.select_action(self.state, epsilon)
            self.state_env_new, r, _ = self.env.update_state(self.state_env_prev, self.actions[action_id])
            self.lidar.update_lidar_state(self.state_env_new, self.env.obstacales, self.grid_size, self.goal_state)

            next_state = self.lidar.state_lidar

            self.state = next_state
            reward += r

            self.env.state_history.append(self.state_env_new)
            self.env.action_history.append(self.actions[action_id].copy())

            frame_id += 1
            print("agent did: " + str(frame_id) + " steps. Reward: " + str(reward) + "action:" + str(self.actions[action_id]))
            self.state_env_prev = self.state_env_new

            if self.is_create_gif:
                self.env.animate_state_history(frame_id=frame_id)

        if self.is_create_gif:
            self.create_gif()

        # Show final plot states
        self.plot_learning_history(self.learning_history)
        print("Agent reward of the episod:", reward)

    def create_gif(self, save_path="frames", output_gif="animation.gif", duration=0.3, loop=0):
        print("Creating gif")
        """Combines saved frames into a GIF."""
        images = []
        for file in sorted(os.listdir(save_path)):
            if file.endswith(".png"):
                print(file)
                images.append(imageio.imread(os.path.join(save_path, file)))
        if images:
            imageio.mimsave(output_gif, images, duration=duration, loop=0)
            print(f"GIF saved as {output_gif}")


if __name__ == "__main__":
    node = DNQSolver()
    node.learnong_dqn()
    node.run_agent_dqn()

