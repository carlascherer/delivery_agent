import numpy as np
import random
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, environment, epsilon=0.1, learning_rate = 0.1, discount_factor=0.6, min_epsilon=0.01, decay_rate=0.995):
        self.env = environment
        # self.q_table = np.zeros((environment.size, environment.size, 4))  # inicialize Q-Table
        self.q_table = np.full((environment.size, environment.size, 4), 0.1)
        self.epsilon = epsilon # inicialized with 1 and then tested with lower values
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.total_rewards_list = []

    def choose_action(self, state):
        # logic to choose the action based on ε-greedy policy
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold >= self.epsilon:  # exploitation: choose the best known action
            action = np.argmax(self.q_table[state])
        else:
            action = random.randint(0, 3)  # exploration: choose random action (4 possible actions)
        return action

    def update_q_table(self, state, action, reward, new_state):
        # logic to update the Q-table using Bellman equation
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[new_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state][action] = new_q

    def train(self, episodes, grid_search_mode=True):
        # logic for the agent training loop
        total_rewards = 0
        for episode in range(episodes):
            state = self.env.reset()  # initialize a new episode and get the inicial state
            done = False
            episode_rewards = 0

            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)  # do the action and receive the new state and reward
                self.update_q_table(state, action, reward, new_state)

                # visualize the agent location
                temp_grid = np.copy(self.env.grid)
                self.env.display(temp_grid, wait=0.5)

                state = new_state
                # accumulate rewards
                episode_rewards += reward

            # register episode performance
            total_rewards += episode_rewards
            self.total_rewards_list.append(total_rewards)
            print(f"Episode {episode + 1}/{episodes} - Rewards: {episode_rewards}, Total Rewards: {total_rewards}")

            # reduce epsilon (exploration rate)
            self.epsilon = max(self.min_epsilon, self.epsilon * np.exp(-self.decay_rate * episode))

        # visualize the evolution of the agent performance
        if not grid_search_mode:
            plt.plot(range(1, episodes+1), self.total_rewards_list)
            plt.xlabel('Episodes')
            plt.ylabel('Total Rewards')
            plt.title('Evolution of the Agent Performance')
            plt.show()

        # save Q-table in a file
        np.save('q_table_final.npy', self.q_table)