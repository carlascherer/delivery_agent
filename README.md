# Projeto de Aprendizado por Reforco para Entrega de Pacotes
Este projeto implementa um agente de aprendizado por reforco que opera em um ambiente de grade 2D.

## Estrutura do Projeto
- DeliveryGrid.py: Define o ambiente de grade 2D.
- DeliveryAgent.py: Contem a implementacao do agente de aprendizado por reforco.
- main.py: Arquivo raiz de execucao.


## Instalação
Certifique-se de ter o Python 3.8+ instalado em sua maquina. Voce pode instalar as dependencias necessárias com o seguinte comando:

```bash
pip install -r requirements.txt
```

## Configuração
Você pode ajustar o número de episódios de treinamento definindo uma variável de ambiente antes de executar o script principal:

```bash
# Para Windows
set EPISODES=500

# Para Linux e macOS
export EPISODES=500
```

## Como Executar
Para iniciar o treinamento do agente, execute o script main.py com o seguinte comando:

```bash
python main.py
```

## Integrantes
- Anthony Moura
- Bruno Neves
- Bruno Bittencurt
- Carla Scherer

