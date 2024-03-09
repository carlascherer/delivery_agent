import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class DeliveryGrid:
    def __init__(self, size=10, num_obstacles=5, num_deliveries=3, min_distance=3):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.obstacle_locations = []
        self.delivery_locations = []
        self.place_objects(num_obstacles, 1)
        self.place_deliveries(num_deliveries, min_distance)
        self.reset_agent()
        self.delivery_reward = 50
        self.obstacle_penalty = -10
        self.step_penalty = -1
        self.deliveries_made = 0
        self.max_steps = size * 2
        self.num_obstacles = num_obstacles
        self.num_deliveries = num_deliveries
        self.agent_position = []

    def place_objects(self, num_objects, object_type):
        placed_objects = 0
        while placed_objects < num_objects:
            location = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if self.grid[location] == 0:
                self.grid[location] = object_type
                if object_type == 1:
                    self.obstacle_locations.append(location)
                placed_objects += 1

    def place_deliveries(self, num_deliveries, min_distance):
        placed_deliveries = 0
        while placed_deliveries < num_deliveries:
            location = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if self.grid[location] == 0:
                too_close = any(np.linalg.norm(np.array(location) - np.array(other), ord=1) < min_distance
                                for other in self.delivery_locations)
                if not too_close:
                    self.grid[location] = 2
                    self.delivery_locations.append(location)
                    placed_deliveries += 1

    def reset_agent(self):
        # Reset the agent's position to an empty cell
        while True:
            location = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if self.grid[location] == 0:
                self.agent_position = location
                break

    def reset(self):
        # Reset the environment state for a new episode
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.place_objects(self.num_obstacles, 1)
        self.place_deliveries(self.num_deliveries, 3)
        self.reset_agent()
        self.deliveries_made = 0
        self.max_steps = self.size * 2
        return self.agent_position

    def step(self, action):
        # Perform the given action and return the new state, reward, and done status
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = directions[action]
        new_position = (self.agent_position[0] + move[0], self.agent_position[1] + move[1])
        reward = self.step_penalty
        done = False

        if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size:
            if self.grid[new_position] == 1:  # Obstacle
                reward = self.obstacle_penalty
            elif self.grid[new_position] == 2:  # Delivery Point
                if new_position in self.delivery_locations:
                    reward = self.delivery_reward
                    self.deliveries_made += 1
                    self.delivery_locations.remove(new_position)
                    self.grid[new_position] = 0
                    if self.deliveries_made == self.num_deliveries:
                        done = True
            self.agent_position = new_position
        else:
            reward = self.obstacle_penalty  # Hit a wall

        self.max_steps -= 1
        if self.max_steps <= 0:
            done = True

        return self.agent_position, reward, done

    def check_delivery(self, position):
        # Verifica se a posição é um ponto de entrega e se a entrega ainda não foi feita
        return position in self.delivery_locations and self.grid[position] == 2
    @staticmethod
    def plot_grid(grid, agent_path, obstacle_locations, delivery_locations, size):
        fig, ax = plt.subplots(figsize=(size, size))       # Plot the grid based on the current state
        ax.imshow(grid, cmap='gray', interpolation='nearest')

        # Plot obstacles
        for obstacle in obstacle_locations:
            ax.scatter(obstacle[1], obstacle[0], marker='s', color='navy', s=100, label='Obstacle')

        # Plot delivery points
        for delivery in delivery_locations:
            ax.scatter(delivery[1], delivery[0], marker='o', color='green', s=100, label='Delivery Point')

        # Plot agent's path
        for step, position in enumerate(agent_path):
            ax.text(position[1], position[0], str(step), ha='center', va='center', color='white')

        # Draw arrows between steps to indicate the path taken
        for i in range(1, len(agent_path)):
            start = agent_path[i - 1]
            end = agent_path[i]
            ax.arrow(start[1], start[0], end[1] - start[1], end[0] - start[0], head_width=0.5, head_length=0.5, fc='yellow', ec='yellow')

        # Custom legend for the plot
        custom_lines = [Line2D([0], [0], color='navy', marker='s', lw=0, markersize=10),
                        Line2D([0], [0], color='green', marker='o', lw=0, markersize=10),
                        Line2D([0], [0], color='yellow', lw=4)]

        ax.legend(custom_lines, ['Obstacle', 'Delivery Point', 'Agent Path'], loc='upper left')

        # Grid and labels
        ax.set_xticks(np.arange(size))
        ax.set_yticks(np.arange(size))
        ax.set_xticklabels(np.arange(1, size+1))
        ax.set_yticklabels(np.arange(1, size+1))
        ax.xaxis.tick_top()

        plt.grid()
        plt.show()


    # def step(self, action):
    #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    #     move = directions[action]
    #     new_position = (self.agent_position[0] + move[0], self.agent_position[1] + move[1])
    #     reward = self.step_penalty
    #     done = False
    #
    #     if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size:
    #         if self.grid[new_position] == 1:  # Obstacle
    #             reward = self.obstacle_penalty
    #         elif self.grid[new_position] == 2:  # Delivery Point
    #             if new_position in self.delivery_locations:
    #                 reward = self.delivery_reward
    #                 self.deliveries_made += 1
    #                 self.delivery_locations.remove(new_position)
    #                 self.grid[new_position] = 0
    #                 if self.deliveries_made == self.num_deliveries:
    #                     done = True
    #         self.agent_position = new_position
    #     else:
    #         reward = self.obstacle_penalty  # Hit a wall
    #
    #     self.max_steps -= 1
    #     if self.max_steps <= 0:
    #         done = True
    #
    #     return self.agent_position, reward, done
    #
    # def is_obstacle(self, position):
    #     x, y = position
    #     return self.grid[x][y] == 1
    #
    # def is_delivery_point(self, position):
    #     x, y = position
    #     return self.grid[x][y] == 2
    #
    # def check_all_deliveries_made(self):
    #     return self.deliveries_made == self.num_deliveries
    #
    # def reset(self):
    #     self.grid = np.zeros((self.size, self.size))
    #     self.place_objects(self.num_obstacles, 1)
    #     self.place_objects(self.num_deliveries, 2)
    #     self.reset_agent()
    #     self.deliveries_made = 0
    #     self.max_steps = self.size * 1
    #     return self.agent_position
    #
    # def place_objects(self, num_objects, object_type):
    #     for _ in range(num_objects):
    #         placed = False
    #         while not placed:
    #             location = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
    #             if self.grid[location] == 0:
    #                 self.grid[location] = object_type
    #                 if object_type == 1:
    #                     self.obstacle_locations.append(location)
    #                 placed = True
    #
    # # def place_objects(self, num_objects, object_type):
    # #     for _ in range(num_objects):
    # #         while True:
    # #             location = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
    # #             if self.grid[location[0]][location[1]] == 0:
    # #                 self.grid[location[0]][location[1]] = object_type
    # #
    # #                 # Armazene a localização com base no tipo de objeto
    # #                 if object_type == 1:
    # #                     self.obstacle_locations.append(location)
    # #                 elif object_type == 2:
    # #                     self.delivery_locations.append(location)
    # #
    # #                 break
    #
    # def place_deliveries(self, num_deliveries, min_distance):
    #     placed_deliveries = 0
    #     while placed_deliveries < num_deliveries:
    #         location = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
    #         if self.grid[location] == 0 and all(
    #                 np.linalg.norm(np.array(location) - np.array(delivery), ord=1) >= min_distance
    #                 for delivery in self.delivery_locations
    #         ):
    #             self.grid[location] = 2  # 2 represents a delivery point
    #             self.delivery_locations.append(location)
    #             placed_deliveries += 1
    #
    # def reset_agent(self):
    #     placed = False
    #     while not placed:
    #         location = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
    #         if self.grid[location] == 0:
    #             self.agent_position = location
    #             placed = True
    #
    # def reset(self):
    #     self.grid = np.zeros((self.size, self.size), dtype=int)
    #     self.place_objects(self.num_obstacles, 1)
    #     self.place_deliveries(self.num_deliveries, 3)
    #     self.reset_agent()
    #     self.deliveries_made = 0
    #     self.max_steps = self.size * 2
    #     return self.agent_position
    #
    # def check_delivery(self, position):
    #     y, x = position
    #     if position in self.delivery_locations:
    #         return True
    #     return False
    #
    # @staticmethod
    # def plot_grid(grid, agent_path, obstacle_locations, delivery_locations, size):
    #     fig, ax = plt.subplots(figsize=(size, size))
    #     ax.imshow(grid, cmap='gray', interpolation='nearest')
    #
    #     # Marcar a posição inicial do agente
    #     if agent_path:
    #         start_pos = agent_path[0]
    #         ax.text(start_pos[1], start_pos[0], 'Start', ha='center', va='center', color='white')
    #
    #     # Marcar obstáculos
    #     for obstacle in obstacle_locations:
    #         ax.scatter(obstacle[1], obstacle[0], marker='s', color='navy', s=100, label='Obstacle')
    #
    #     # Marcar entregas com sucesso e falhas
    #     for delivery in delivery_locations:
    #         if delivery in agent_path:
    #             ax.scatter(delivery[1], delivery[0], marker='*', color='lime', s=100, label='Successful Delivery')
    #         else:
    #             ax.scatter(delivery[1], delivery[0], marker='X', color='red', s=100, label='Failed Delivery')
    #
    #     # Plotar as pegadas do agente
    #     for step, position in enumerate(agent_path):
    #         if position != start_pos:  # Evitar sobreposição com a posição inicial
    #             ax.text(position[1], position[0], str(step), ha='center', va='center', color='gray')
    #
    #     # Desenhar linha do caminho percorrido pelo agente
    #     for i in range(1, len(agent_path)):
    #         ax.plot([agent_path[i - 1][1], agent_path[i][1]], [agent_path[i - 1][0], agent_path[i][0]],
    #                 color='yellow', linewidth=2)
    #
    #     # Adicionando a legenda
    #     legend_elements = [
    #         Line2D([0], [0], marker='s', color='w', label='Obstacle', markerfacecolor='navy', markersize=10),
    #         Line2D([0], [0], marker='*', color='w', label='Successful Delivery', markerfacecolor='lime', markersize=10),
    #         Line2D([0], [0], marker='X', color='w', label='Failed Delivery', markerfacecolor='red', markersize=10),
    #         Line2D([0], [0], color='yellow', lw=2, label='Agent Path')]
    #     # Create a legend
    #     ax.legend(handles=legend_elements, loc='upper left')
    #
    #     # Plot the agent's path with arrows
    #     for i in range(1, len(agent_path)):
    #         ax.annotate('', xy=agent_path[i], xytext=agent_path[i - 1],
    #                     arrowprops=dict(facecolor='yellow', shrink=0.05),
    #                     )
    #
    #     # Setting grid labels and titles
    #     ax.set_xticks(np.arange(size))
    #     ax.set_yticks(np.arange(size))
    #     ax.set_xticklabels(np.arange(size))
    #     ax.set_yticklabels(np.arange(size))
    #     ax.set_title("Delivery Grid Path")
    #
    #     # Turn on the grid
    #     ax.grid(True)
    #
    #     # Show the plot
    #     plt.show()
    #
    # def mark_delivery(self, position):
    #     # Marca no grid que uma entrega foi feita
    #     y, x = position
    #     self.grid[x][y] = 0
