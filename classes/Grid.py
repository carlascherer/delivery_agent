import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import matplotlib.patches as mpatches
import time

class Grid:
    # def __init__(self, size=15, num_obstacles=7, num_deliveries=5, delivery_reward=10, obstacle_penalty=-5, step_penalty=-1, deliveries_made=0):#min_distance=3
    def __init__(self, size=5, num_obstacles=3, num_deliveries=1, delivery_reward=10, obstacle_penalty=-5, step_penalty=-1, deliveries_made=0):#min_distance=3
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.num_obstacles = num_obstacles
        self.num_deliveries = num_deliveries
        self.place_objects(num_obstacles, 1)  # place obstacles
        self.place_objects(num_deliveries, 2)  # place delivery points
        self.reset_agent()  # place the agent
        self.delivery_reward = delivery_reward # reward for delivering the package
        self.obstacle_penalty = obstacle_penalty # penalty for hitting an obstacle
        self.step_penalty = step_penalty # penalty for every step
        self.deliveries_made = deliveries_made # counter of the number of deliveries made
        self.max_steps = self.size * 10 # max number of steps in each episode

        # 0: empty space
        # 1: obstacle
        # 2: delivery point
        # 3: agent location

    def place_objects(self, count, object_type):
        placed = 0
        while placed < count:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[x, y] == 0:
                self.grid[x, y] = object_type
                placed += 1

    def reset_agent(self):
        while True:
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[x, y] == 0:
                self.agent_position = (x, y)
                break


    def display(self, temp_grid):
        # logic to visualize the grid and agent position
        clear_output(wait=True)
        plt.figure(figsize=(5, 5))
        title = 'Reinforcement Learning - Delivery Agent'
        plt.title(title, fontdict={'fontsize': 13, 'fontweight': 'bold'})
        plt.imshow(temp_grid, cmap='magma', interpolation='nearest')

        # annotate obstacles, delivery points and agent position
        for y in range(temp_grid.shape[0]):
            for x in range(temp_grid.shape[1]):
                text = ''
                color = 'white'
                if temp_grid[y, x] == 1:
                    text = 'Obstacle'
                elif temp_grid[y, x] == 2:
                    text = 'Delivery Point'
                elif temp_grid[y, x] == 3:
                    text = 'Agent Location'
                    color = 'black'
                if text:
                    plt.text(x, y, text, ha='center', va='center', color=color, fontsize=6)

        plt.axis('off')

        plt.show()
        time.sleep(0.5)


    def step(self, action):
        # calculate new position based on the action
        new_position = self.calculate_new_position(action)

        # verify grid limits
        if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size:
            # verify if new position is a delivery point or an obstacle
            if self.is_obstacle(new_position):
                reward = self.obstacle_penalty
                done = False
            elif self.is_delivery_point(new_position):
                reward = self.delivery_reward
                self.grid[new_position] = 0  # remove delivery point
                self.deliveries_made += 1
                done = self.check_all_deliveries_made()
            else:
                reward = self.step_penalty
                done = False
            # update the agent position if new position is not an obstacle
            if not self.is_obstacle(new_position):
                self.agent_position = new_position  # move agent
        else:
            # if new position is out of grid limits, penalize and dont move
            reward = self.step_penalty
            done = False

        # verify if max steps number was reached
        self.max_steps -= 1
        if self.max_steps <= 0:
            done = True

        # new state is the agent position
        new_state = self.agent_position

        return new_state, reward, done


    def calculate_new_position(self, action):
        # logic to calculate new position based on action

        # 0: left
        # 1: right
        # 2: down
        # 3: up

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = directions[action]
        return (self.agent_position[0] + move[0], self.agent_position[1] + move[1])


    def is_delivery_point(self, position):
        # verify if new position is a delivery point
        x, y = position
        return self.grid[x][y] == 2


    def is_obstacle(self, position):
        # verify if new position is an obstacle
        x, y = position
        return self.grid[x][y] == 1


    def check_all_deliveries_made(self):
        # Verifique se todos os pacotes foram entregues
        return self.deliveries_made == self.num_deliveries


    def reset(self):
        # reset environment to initial state
        self.grid = np.zeros((self.size, self.size))
        self.place_objects(self.num_obstacles, 1)
        self.place_objects(self.num_deliveries, 2)
        self.reset_agent()
        self.deliveries_made = 0
        self.max_steps = self.size * 10  # reset steps counter
        return self.agent_position
