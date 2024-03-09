from classes.Agent import Agent
from classes.Grid import Grid
import numpy as np


def main():
    # define number of episodes
    episodes = 100

    # instantiate environment
    size = 10
    num_obstacles = 5
    num_deliveries = 4
    delivery_reward = 10
    obstacle_penalty = -5
    step_penalty = -1
    deliveries_made = 0

    environment = Grid(size, num_obstacles, num_deliveries, delivery_reward, obstacle_penalty, step_penalty, deliveries_made)

    # # instantiate agent
    # learning_rate = 0.01
    # discount_factor = 0.95
    # epsilon = 1.0
    # min_epsilon = 0.01
    # decay_rate = 0.01

    # agent = Agent(environment, epsilon, learning_rate, discount_factor, min_epsilon, decay_rate)

    print("Initializing grid search for training")

    learning_rates = [0.1, 0.01, 0.001]
    discount_factors = [0.8, 0.7, 0.6]
    epsilons = [1.0, 0.5, 0.1]
    min_epsilons = [0.1, 0.01, 0.001]
    decay_rates = [0.01, 0.001, 0.1]

    best_reward = float('-inf')
    best_params = {}

    for lr in learning_rates:
        for df in discount_factors:
            for eps in epsilons:
                for min_eps in min_epsilons:
                    for dr in decay_rates:
                        agent = Agent(environment, learning_rate=lr, discount_factor=df, epsilon=eps, min_epsilon=min_eps, decay_rate=dr)
                        agent.train(episodes, grid_search_mode=True)
                        total_reward = sum(agent.total_rewards_list)

                        if total_reward > best_reward:
                            best_reward = total_reward
                            best_params = {'learning_rate': lr, 'discount_factor': df, 'epsilon': eps, 'min_epsilon': min_eps, 'decay_rate': dr}

    print("Best params:")
    print(best_params)
    print("Best reward:", best_reward)

    print("Grid search for training is done!")

    # training agent with best params
    learning_rate = best_params['learning_rate']
    discount_factor = best_params['discount_factor']
    epsilon = best_params['epsilon']
    min_epsilon = best_params['min_epsilon']
    decay_rate = best_params['decay_rate']

    agent = Agent(environment, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon, min_epsilon=min_epsilon, decay_rate=decay_rate)
    agent.train(episodes, grid_search_mode=False)

    # save Q-table in a file
    np.save('q_table.npy', agent.q_table)

if __name__ == "__main__":
    main()