import os

from classes.DeliveryAgent import DeliveryAgent
from classes.DeliveryGrid import DeliveryGrid


def main():
    env = DeliveryGrid()

    # Parâmetros do agente
    learning_rate = 0.005
    discount_factor = 0.95
    epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01

    # Criar o agente
    agent = DeliveryAgent(env, learning_rate, discount_factor, epsilon, min_epsilon, decay_rate)

    # Parametros de execucao
    episodes = int(os.getenv('EPISODES', '300'))
    view_episode = 100

    # Treinamento do agente
    print("Iniciando o treinamento para {} episódios...".format(episodes))
    agent.train(episodes, view_episode)
    print("Treinamento concluído.")


if __name__ == "__main__":
    main()
