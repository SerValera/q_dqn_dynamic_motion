import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

# Define the environment class for the MDP
class GridEnvironment:
    def __init__(self, grid_size=(4, 4), init_state = [0, 0, 0], goal = [3, 2], obstacales_list=[]):
        self.init_state = init_state
        self.state = self.init_state  # Initial state [x, y, yaw in degrees]
        self.obstacales = obstacales_list
        self.is_show_all_history_states = True

        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)  # 0 indicates free cell, 1 indicates obstacle (none for now)

        self.prev_state = self.init_state
        self.goal = goal  # Goal state at the bottom-right corner
        self.state_history = [self.init_state]
        self.action_history = []
        self.is_move_obst = False

        self.frames = []
        
        self.reward = 0
        self.grid[self.init_state[0], self.init_state[1]] = 1 # agent

        self.actions = [[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]

        # for obs in self.obstacales:
        #     self.grid[obs[0], obs[1]] = 2 # obst

        self.grid[self.goal[0], self.goal[1]] = 3 # goal

    def move_obstacles(self, state):
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left
        move_times = random.randint(1, 2)
        # move_times = 1
        
        new_obstacles = self.obstacales.copy()
        for step in range(3):  # Maximum steps an obstacle might move
            for i, (x, y) in enumerate(new_obstacles):
                move_times = random.randint(1, 2)
                if move_times == 1:
                    dx, dy = random.choice(directions)

                    new_x = x + dx
                    new_y = y + dy

                    if new_x < 0 or new_x == self.grid_size[0] or new_y < 0 or new_y == self.grid_size[1]:
                        new_x = x
                        new_y = y

                    if new_x == state[0] or new_x == state[1]:
                        new_x = x
                        new_y = y

                    if new_x == self.goal[0] or new_y == self.goal[1]:
                        new_x = x
                        new_y = y

                    new_obstacles[i] = [new_x, new_y]
        self.obstacales = new_obstacles


    def generate_random_obstacles(self, num_obstacles):
        obstacles = []
        while len(obstacles) < num_obstacles:
            obstacle = [np.random.randint(0, self.grid_size[0]-1), np.random.randint(0, self.grid_size[1]-1)]
            if obstacle not in [tuple(self.init_state), tuple(self.goal)]:
                obstacles.append(obstacle)
        self.obstacales = obstacles
        print("env. obstacales: ", self.obstacales)
        # for obs in self.obstacales:
        #     self.grid[obs[0], obs[1]] = 2 # obst
        return self.obstacales 

    def reset(self):
        """Resets the environment to the given start state."""
        self.state = self.init_state
        return self.state

    def run(self):
        actions = [self.actions[5], self.actions[5], self.actions[0]]
        num_steps = len(actions)

        for i in range(num_steps):
            self.step(actions[i])
            self.state_history.append(self.state.copy())
            self.action_history.append(actions[i].copy())
            # print("reward", self.reward)

    def rotated(self, dx,dy,yaw):
        x = dx * math.cos(yaw*math.pi/180) - dy * math.sin(yaw*math.pi/180)
        y = dx * math.sin(yaw*math.pi/180) + dy * math.cos(yaw*math.pi/180)
        if x > 0.6:
            x = 1
        if x < -0.6:
            x = -1
        if y > 0.6:
            y = 1
        if y < -0.6:
            y = -1
        return x, y

    def update_state(self, current_state, action):
        # Extract current state and action
        x, y, yaw = current_state
        dx, dy, dyaw = action
        reward = 0
        
        # Movement deltas based on yaw
        if yaw == 0:        # Facing right
            move_x, move_y = dx, dy
        elif yaw == 90:     # Facing up
            move_x, move_y = -dy, dx
        elif yaw == 180:    # Facing left
            move_x, move_y = -dx, -dy
        elif yaw == 270:    # Facing down
            move_x, move_y = dy, -dx

        elif yaw == 45:     # Diagonal up-right
            move_x, move_y = self.rotated(dx,dy,yaw)
        elif yaw == 135:    # Diagonal up-left
            move_x, move_y = self.rotated(dx,dy,yaw)
        elif yaw == 225:    # Diagonal down-left
            move_x, move_y = self.rotated(dx,dy,yaw)
        elif yaw == 315:    # Diagonal down-right
            move_x, move_y = self.rotated(dx,dy,yaw)
        
        # Update state
        new_x = x + move_x
        new_y = y + move_y
        new_yaw = (yaw + dyaw) % 360  # Keep yaw between 0 and 360 degrees

        if [new_x, new_y, new_yaw] != current_state:
            # Step reward/cost
            dist = - round(np.sqrt((current_state[0] - new_x)**2 + (current_state[1] - new_y)**2), 2) / 10

            # Rotation reward/cost
            # theta1, theta2 = current_state[2], new_yaw
            # rot = round(-abs(np.arctan2(np.sin(np.deg2rad(theta2) - np.deg2rad(theta1)), np.cos(np.deg2rad(theta2) - np.deg2rad(theta1)))) / 7, 1)
            rot = 0
            reward = dist + rot

        # If collision, stat same point.
        for obst in self.obstacales: 
            if [new_x, new_y] == obst:
                reward = -1
                new_x = x
                new_y = y

        # check bounds
        if new_x < 0 or new_x == self.grid_size[0] or new_y < 0 or new_y == self.grid_size[1]:
            reward = -1
            new_x = x
            new_y = y

        if [new_x, new_y] == self.goal:
            reward = 10

        if self.is_move_obst:
            self.move_obstacles([new_x, new_y])
        
        return [int(new_x), int(new_y), new_yaw], reward

    def step(self, action):
        """Takes an action and updates the state.
        
        Actions: [x, y, yaw] where:
        - x: move forward (-1), stay (0), or move back (1)
        - y: move left (-1), stay (0), or move right (1)
        - yaw: turn left (-45), stay (0), or turn right (45)
        """        
        # print("current state", self.state)
        self.grid[int(self.state[0]), int(self.state[1])] = 1
        self.state = self.update_state(self.state, action)
        self.prev_state = self.state
        return self.state, self._get_reward(), self._is_done()

    def _get_reward(self):
        """Returns the reward for the current state."""

        # Goal
        if self.state[:2] == self.goal: 
            self.reward += 10
        
        # Colllision check
        # for obst in self.obstacales: 
        #     if self.state[:2] == obst:
        #         print('colision')
        #         self.reward -= 1
        
        if self.state != self.prev_state:
            # Step reward/cost
            dist = - np.sqrt((self.prev_state[0] - self.state[0])**2 + (self.prev_state[1] - self.state[1])**2)

            # Rotation reward/cost
            theta1, theta2 = self.prev_state[2], self.state[2]
            rot = - np.arctan2(np.sin(np.deg2rad(theta2) - np.deg2rad(theta1)), np.cos(np.deg2rad(theta2) - np.deg2rad(theta1)))
            rot = 0

            return dist + rot

    def _is_done(self):
        """Checks if the current state is the goal state."""
        return self.state[:2] == self.goal

    def render(self):
        """Renders the grid and plots the drone's current state and goal."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap='gray', origin='upper')
        plt.scatter(self.state[1], self.state[0], c='red', s=100, label='Drone')  # Drone position
        plt.scatter(self.goal[1], self.goal[0], c='green', s=100, label='Goal')  # Goal position
        plt.arrow(
            self.state[1], self.state[0],
            0.5 * np.cos(np.deg2rad(self.state[2])),
            -0.5 * np.sin(np.deg2rad(self.state[2])),
            head_width=0.3, head_length=0.3, fc='red', ec='red'
        )
        plt.xticks(np.arange(0, self.grid_size[1], 1))
        plt.yticks(np.arange(0, self.grid_size[0], 1))
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_state_history(self):
        """Plots the history of states on the grid with arrows indicating direction."""
        history = np.array(self.state_history)
        x_history, y_history, yaw_history = history[:, 0], history[:, 1], history[:, 2]

        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap='gray', origin='upper')

        if self.is_show_all_history_states:
            plt.scatter(y_history, x_history, c='blue', s=50, label='State History')

        plt.scatter(y_history[len(y_history) - 1], x_history[len(x_history) - 1], c='red', s=50, label='Agent current state')
                    
        plt.scatter(self.goal[1], self.goal[0], c='green', s=100, label='Goal')  # Goal position

        # Add arrows to indicate the direction of movement at each state
        # for i in range(len(x_history)):
        #     plt.arrow(
        #         y_history[i], x_history[i],
        #         0.5 * np.cos(np.deg2rad(yaw_history[i])),
        #         -0.5 * np.sin(np.deg2rad(yaw_history[i])),
        #         head_width=0.3, head_length=0.3, fc='red', ec='red'
        #     )

        for obst in self.obstacales:
            plt.scatter(obst[1], obst[0], c='gray', s=100)

        plt.xticks(np.arange(0, self.grid_size[1], 1))
        plt.yticks(np.arange(0, self.grid_size[0], 1))
        plt.grid(True)
        # plt.legend()
        plt.show()

    def animate_state_history(self, save_path="frames", frame_id=0):
        """Plots the history of states on the grid with arrows indicating direction and saves frames."""
        history = np.array(self.state_history)
        x_history, y_history, yaw_history = history[:, 0], history[:, 1], history[:, 2]

        plt.figure(figsize=(8, 8))
        plt.imshow(self.grid, cmap='gray', origin='upper')

        if self.is_show_all_history_states:
            plt.scatter(y_history, x_history, c='blue', s=50, label='State History')
        
        plt.scatter(y_history[len(y_history) - 1], x_history[len(x_history) - 1], c='red', s=100, label='State History')
        plt.scatter(self.goal[1], self.goal[0], c='green', s=100, label='Goal')  # Goal position

        for obst in self.obstacales:
            plt.scatter(obst[1], obst[0], c='gray', s=100)

        plt.xticks(np.arange(0, self.grid_size[1], 1))
        plt.yticks(np.arange(0, self.grid_size[0], 1))
        plt.grid(True)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        frame_filename = os.path.join(save_path, f"frame_{frame_id:03d}.png")
        plt.savefig(frame_filename)  # Save frame
        plt.close()