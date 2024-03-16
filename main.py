from classes.Agent import Agent
from classes.Grid import Grid
import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(2147483648)

    # define number of episodes
    episodes = 100

    # instantiate environment
    size = 5
    num_obstacles = 3
    num_deliveries = 2
    delivery_reward = 20
    obstacle_penalty = -5
    step_penalty = -0.1
    deliveries_made = 0

    environment = Grid(size, num_obstacles, num_deliveries, delivery_reward, obstacle_penalty, step_penalty, deliveries_made)


    print("Initializing grid search for training")

    learning_rates = [0.1, 0.01, 0.001]
    discount_factors = [0.8, 0.7, 0.6]
    epsilons = [1.0, 0.8, 0.6]
    min_epsilons = [0.5, 0.4, 0.3]
    decay_rates = [0.00001]

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

    print("Grid search for training is done!")

    # training agent with best params
    learning_rate = best_params['learning_rate']
    discount_factor = best_params['discount_factor']
    epsilon = best_params['epsilon']
    min_epsilon = best_params['min_epsilon']
    decay_rate = best_params['decay_rate']

    episodes=10000

    agent = Agent(environment, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon, min_epsilon=min_epsilon, decay_rate=decay_rate)
    agent.train(episodes, grid_search_mode=False)

    # save Q-table in a file
    np.save('q_table.npy', agent.q_table)

    print("Best params:")
    print(best_params)
    print("Best reward:", best_reward)

    # test agent performance after training
    state = environment.reset()
    done = False
    plt.ion()
    fig, ax = plt.subplots()
    while not done:
        ax.clear()
        action = agent.choose_action(state)
        state, reward, done = environment.step(action)

        temp_grid = np.copy(environment.grid)
        y, x = state
        temp_grid[y, x] = 3  # points the current agent location
        environment.display(temp_grid)  # visualize grid state

    plt.ioff()
    


if __name__ == "__main__":
    main()