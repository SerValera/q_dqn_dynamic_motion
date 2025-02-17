import numpy as np
import matplotlib.pyplot as plt
import math

class LidarObservation:
    def __init__(self, beem_number=8):
        self.state_lidar = np.zeros(beem_number + 1) 
        self.distance_to_goal = 9
        self.state = [0, 0, 0]
        self.lidar_configuration = [
            [[0, 1], [0, 2]], 
            [[-1, 1], [-2, 2]],
            [[-1, 0], [-2, 0]],
            [[-1, -1], [-2, -2]],
            [[0, -1], [0, -2]],
            [[1, -1], [2, -2]],
            [[1, 0], [2, 0]],
            [[1, 1], [2, 2]]
        ]
        self.grid_lidar = np.zeros((5, 5))

    def update_lidar_grid(self, robot_state, dist_to_goal, obstacle_list):
        self.state_lidar[8] = dist_to_goal    
        for idx, config in enumerate(self.lidar_configuration):
            for i in range(len(config)):
                cx = config[i][0] + robot_state[0]
                cy = config[i][1] + robot_state[1]

                print('conf', cx, cy)
                for obs in obstacle_list:
                    ox = obs[0]
                    oy = obs[1]

                    print(ox, oy)

                    if cx == ox and cy == oy:
                        self.state_lidar[idx] = i+1
        # print(self.state_lidar)

    def update_lidar_state(self, robot_state, obstacle_list, grid_size, goal):
        self.state_lidar = np.zeros(9) 
        self.state_lidar[8] = math.sqrt((robot_state[0] - goal[0])**2 + (robot_state[1] - goal[1])**2)

        for idx, config in enumerate(self.lidar_configuration):
            for i in range(len(config)):
                beem_x = config[i][0] + robot_state[0]
                beem_y = config[i][1] + robot_state[1]

                if beem_x < 0 or beem_x == grid_size[0] or beem_y < 0 or beem_y == grid_size[1]:
                    if self.state_lidar[idx] != 1:
                            self.state_lidar[idx] = i+1

                for obs in obstacle_list:
                    obst_x = obs[0]
                    obst_y = obs[1]

                    if beem_x == obst_x and beem_y == obst_y:

                        if self.state_lidar[idx] != 1:
                            self.state_lidar[idx] = i+1
        # print(self.state_lidar)
    

    def show_lidar_grid(self):
        plt.figure(figsize=(6, 6))
        
        for idx, config in enumerate(self.lidar_configuration):
            for point in config:
                beem_x = int(point[0]) + 2 
                beem_y = int(point[1]) + 2
                
                if 0 <= beem_x < self.grid_lidar.shape[0] and 0 <= beem_y < self.grid_lidar.shape[1]:
                    self.grid_lidar[beem_x, beem_y] = self.state_lidar[idx]
        
        for i in range(self.grid_lidar.shape[0]):
            for j in range(self.grid_lidar.shape[1]):
                plt.text(j, i, f'{self.grid_lidar[i, j]:.1f}', ha='center', va='center', color='black')
        
        plt.imshow(self.grid_lidar, origin='upper', interpolation='nearest')
        
        plt.title("Lidar Grid Visualization")
        plt.colorbar(label='Lidar Intensity')
        plt.show()
