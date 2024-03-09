import numpy as np
import random
from classes.DeliveryGrid import DeliveryGrid


class DeliveryAgent:
    def __init__(self, environment, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, min_epsilon=0.01,
                 decay_rate=0.01):
        self.env = environment
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.q_table = np.full((environment.size, environment.size, 4), 0.1)
        self.episode_rewards_list = []
        self.agent_path = []

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, new_state):
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[new_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                reward + self.discount_factor * max_future_q)
        self.q_table[state][action] = new_q

    def train(self, episodes, display_episode=None):
        total_rewards = 0
        # episode_rewards_list = []
        deliveries_attempted = []

        for episode in range(episodes):
            done = False
            episode_rewards = 0
            # episode_path = []
            deliveries_made = []
            self.agent_path.clear()  # Limpe o caminho no início de cada episódio
            state = self.env.reset()

            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                episode_rewards += reward
                # episode_path.append(state)
                self.agent_path.append(state)  # Adicione o estado atual ao caminho do agente
                if self.env.check_delivery(state):
                    deliveries_made.append(state)

            total_rewards += episode_rewards

            print(
                f"Episódio {episode + 1}/{episodes} - Recompensa: {episode_rewards}, Total Recompensas: {total_rewards}")

            # Atualizar epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * np.exp(-self.decay_rate * episode))

            # Se for o episódio de exibição, mostre o grid com o caminho percorrido
            if episode == display_episode:
                DeliveryGrid.plot_grid(
                    self.env.grid,
                    self.agent_path,
                    self.env.obstacle_locations,
                    self.env.delivery_locations,
                    self.env.size
                )
            np.save('episode_rewards.npy', total_rewards)
