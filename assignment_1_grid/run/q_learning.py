import numpy as np
import random
from env import *
import imageio
import os

class QSolver:
    def __init__(self):
        # Env setup
        self.grid_size = (10, 20)
        self.init_state = [0, 0, 0]
        self.goal_state = [9, 8]

        # Obst setup
        self.obstacales_list = []
        self.env = GridEnvironment(self.grid_size, self.init_state, self.goal_state, self.obstacales_list)
        num_obstacles = 10
        self.obstacales_list = self.env.generate_random_obstacles(num_obstacles)

        # True - Dynamic env
        # False - Static env (deafult)
        self.env.is_move_obst = True

        # to create gif
        self.is_create_gif = True
        self.env.is_show_all_history_states = True
        self.is_plot_steps_history = True
    
        # Action list
        self.actions = [[0, 1, 0], [0, -1, 0], [1, 0, 0],  [-1, 0, 0],  #Forward, Back, Right, Left
                        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]  #FR, FL, BR, BL 
                        ] 

        # Learning setup
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.5  # Probability to take random action
        self.iterations = 1000

        self.Q = np.zeros((self.grid_size[0], self.grid_size[1], len(self.actions)))  # Q-values for each action
        
        # To analyse learning
        self.learning_history = []

        self.k = 0
        self.max_step = 250
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


    def learning(self):
        for episode in range(self.iterations):
            print("episode#", episode, "started")
            self.k = 0
            reward_hist = 0
            state = self.init_state
            while state[:2] != self.goal_state or self.k == self.max_step:
                rn = random.uniform(0, 1)
                if random.uniform(0, 1) < self.epsilon:
                    action_id = random.randint(0, len(self.actions)-1)
                else:
                    action_id = np.argmax(self.Q[state[0], state[1], :])

                state_new, reward = self.env.update_state(state, self.actions[action_id])
                next_state = state_new
                best_next_action = np.argmax(self.Q[next_state[0], next_state[1], :])
                
                self.Q[state[0], state[1], action_id] += self.alpha * (reward + self.gamma * self.Q[next_state[0], next_state[1], best_next_action] - self.Q[state[0], state[1], action_id])
                
                state = next_state
                reward_hist += reward
                self.k += 1
            self.steps_history.append([episode, self.k])
            self.learning_history.append([episode, reward_hist])

    def print_results(self):
        print("Q-Values:")
        print(np.round(self.Q, 2))

    def run_agent(self):
        print("agent run")
        state = self.init_state
        frame_id = 0
        reward = 0

        while state[:2] != self.goal_state:
            action_id = np.argmax(self.Q[state[0], state[1], :])  # Take best action
            next_state, r = self.env.update_state(state, self.actions[action_id])
            state = next_state
            reward += r

            self.env.state_history.append(state)
            self.env.action_history.append(self.actions[action_id].copy())

            frame_id += 1  # Увеличиваем счётчик кадров
            print("agent did: " + str(frame_id) + " steps. Reward: " + str(reward))

            if self.is_create_gif:
                self.env.animate_state_history(frame_id=frame_id)

        if self.is_create_gif:
            self.create_gif()

        if self.is_plot_steps_history:
            self.plot_steps_history(self.steps_history)

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

    def q_vis(self, Q):
        actions = self.actions
        fig, axes = plt.subplots(2, 4, figsize=(8, 4)) 
        fig.suptitle("Q-values for Each Action", fontsize=10)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < len(actions):
                cax = ax.matshow(Q[:, :, i], cmap="coolwarm")
                ax.set_title(f"Action: {actions[i]}", fontsize=12)
            else:
                ax.axis("off")
        fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.1)
        save_dir = os.path.join(os.path.dirname(__file__), "..", "analysis")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "q_table.png")
        plt.savefig(save_path)
        print(f"saved to {save_path}")
        plt.show()
        plt.close()

if __name__ == "__main__":
    node = QSolver()
    node.learning()
    node.q_vis(node.Q)
    node.run_agent()

