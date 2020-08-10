import gym
from keras import models
from keras import layers
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

try:
     path = sys.argv[1]
except:
     path = '.'

#Lendo o arquivo txt de parâmetros
linelist = [line.rstrip('\n') for line in open(path + "/parametros.txt")]

#Parâmetros definidos pelo usuário
GAMMA = float(linelist[1])
EPSILON_DECAY = float(linelist[2])
EPSILON_MIN = float(linelist[3])
LEARNING_RATE = float(linelist[4])
NUMBER_OF_EPISODES = int(linelist[5])
NUMBER_OF_ITERATIONS = int(linelist[6])
PICK_FROM_BUFFER_SIZE = int(linelist[7])

#Parâmetros definidos pelo sistema
BUFFER_LEN = 200000
EPSILON = 1

#Definindo parâmetros da rede neural
netlist = [line.rstrip('\n') for line in open(path + "/network.txt")]
NUMBER_OF_LAYERS = int(netlist[0])
NEURONS_PER_LAYER = []
for i in range(NUMBER_OF_LAYERS):
    NEURONS_PER_LAYER.append(int(netlist[i+1]))
if(len(NEURONS_PER_LAYER)!=NUMBER_OF_LAYERS):
    pf("INCONSISTENCIA NA INSERCAO DE CAMADAS E NEURONIOS POR CAMADA", flush=True)
    exit()

def pf(string, **kwargs):
    if ("flush") in kwargs:
        print(string, **kwargs)
    else:
        print(string, flush= True, **kwargs)


class DQN_Agent:

    def __init__(self, env):
    #Definindo as variáveis
        self.env = env
        self.gamma = GAMMA

        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        self.learning_rate = LEARNING_RATE

        self.replay_buffer = deque(maxlen = BUFFER_LEN)

        self.train_network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.set_weights(self.train_network.get_weights())

        self.episode_num = NUMBER_OF_EPISODES
        self.iteration_num = NUMBER_OF_ITERATIONS
        self.pick_buffer_every = PICK_FROM_BUFFER_SIZE

        #Variáveis de análise
        self.total_rw_per_ep = []
        self.total_steps_per_ep = []

#Modelando a rede neural
    def create_network(self):
        model = models.Sequential()
        
        #Pega o tamanho do espaço de observações do ambiente
        state_shape = self.env.observation_space.shape

        #A rede tem arquitetura escolhida pelo usuário
        for i in range(NUMBER_OF_LAYERS):
            if(i == 0):
                model.add(layers.Dense(NEURONS_PER_LAYER[0], activation='relu', input_shape=state_shape))
            else:
                model.add(layers.Dense(NEURONS_PER_LAYER[i], activation='relu'))
        #O tamanho da output layer é igual ao tamanho do espaço de ações
        model.add(layers.Dense(self.env.action_space.n, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

#Escolhe qual ação tomar(aleatória ou não)
    def greedy_action(self, state):
        #print("ESTADO: ", state)

        #Se atingir o epsilon min, fica nele.
        self.epsilon = max(self.epsilon_min, self.epsilon)

        #Escolhe um número aleatório entre 0 e 1. Se ele for menor do que epsilon, toma uma ação aleatória
        if(np.random.rand(1) < self.epsilon):
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(self.train_network.predict(state)[0])

        #print("ACAO: ", action)
        return action

    def replay_memory(self):
        #Checa se o tamanho atual do buffer é menor do que o tamanho mínimo necessário
        if len(self.replay_buffer) < self.pick_buffer_every:
            return

        #Pega uma amostra randomica do buffer. A amostra possui tamanho "pick_buffer_every"
        samples = random.sample(self.replay_buffer, self.pick_buffer_every)

        states = []
        new_states=[]

        #Itera em samples. Cada sample tem a forma (state, action, reward, new_state, done)
        for sample in samples:
            #Armazena a sample atual nas variáveis
            state, action, reward, new_state, done = sample
            #Adiciona as variáveis nas listas.
            states.append(state)
            new_states.append(new_state)

        #Transforma a lista de estados em um array
        states_array = np.array(states)
        #Dá um reshape, criando uma linha para cada step(para cada sample) e n colunas, uma para cada estado.
        states = states_array.reshape(self.pick_buffer_every, env.observation_space.shape[0])
        #Transforma a lista de estados em um array
        new_states_array = np.array(new_states)
        #Dá um reshape, criando uma linha para cada step(para cada sample) e n colunas, uma para cada estado.
        new_states = new_states_array.reshape(self.pick_buffer_every, env.observation_space.shape[0])

        #Dá um predict na train_network para pegar os Q-values atuais
        targets = self.train_network.predict(states)
        #Dá um predict na target_network para pegar os novos Q-values
        new_qs = self.target_network.predict(new_states)

        i = 0
        for sample in samples:
            state, action, reward, new_state, done = sample
            #target recebe os Q-values antigos da iteração atual
            target = targets[i]
            #Caso tenha terminado, o Q-value referente à ação atual recebe "reward"
            if done:
                target[action] = reward
            #Caso não tenha terminado, recebe a relação do Q-Learning.
            else:
                #new_q recebe o novo Q-value da iteração atual. Note que "max" já passa o maior Q-value.
                new_q = max(new_qs[i])
                #O Target da ação da iteração atual recebe então a relação.
                target[action] = reward + new_q * self.gamma
            i+=1

        #Por fim, treina a train_network com os Q-values atualizados
        self.train_network.fit(states, targets, epochs=1, verbose=0)


    def play(self, current_state, eps):
            reward_sum = 0
            #Começa a posição máxima com um valor baixo. Guardará a posição máxima de cada step
            max_position = -99
            #Itera nos steps
            for i in range(self.iteration_num):
                action = self.greedy_action(current_state)

                #Renderiza a cada 50 episódios
                #if(eps%20 == 0):
                 #   env.render()

                #Agente toma a ação
                new_state, reward, done, _ = env.step(action)
                new_state = new_state.reshape(1, env.observation_space.shape[0])

                #Guarda a posição máxima
                if(new_state[0][0] > max_position):
                    max_position = new_state[0][0]


                #Adiciona os dados do step no buffer
                self.replay_buffer.append([current_state, action, reward, new_state, done])

                #Chama o replay memory, que só é executado quando temos um buffer de tamanho aceitável.
                self.replay_memory()

                #Soma a reward e atualiza o estado atual
                reward_sum += reward
                current_state = new_state

                #Caso tenha concluído no step atual, dá um break no loop
                if done:
                    pf("Episodio: " + str(eps).zfill(3) + " || " + str(i).zfill(3) + " Steps" + " || Reward: " + str(reward_sum).zfill(3) + " || EPSILON: " + '{:.3f}'.format(self.epsilon), flush=True)
                    self.train_network.save_weights(path + '/model_inverted_numerical/model_numerical.h5')
                    break

            #Armazena a reward total e o numero total de steps gastos para, posteriormente, plotar graficos
            self.total_rw_per_ep.append(reward_sum)
            self.total_steps_per_ep.append(i)

            #Copia os pesos da target para a train
            self.target_network.set_weights(self.train_network.get_weights())

            #Decai o epsilon
            self.epsilon -= self.epsilon_decay

    def start(self):
        #Itera nos episódios
        for eps in range(self.episode_num):

            current_state = env.reset().reshape(1, env.observation_space.shape[0])
            self.play(current_state, eps)

            #Plotando resultados
            plot(self.total_rw_per_ep)

def plot(reward_store):
    #Fazendo uma média das rewards
    N = 15
    cumsum, moving_aves = [0], []
    for i, x in enumerate(reward_store, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    #Reward
    plt.figure(num=None, figsize=(20, 12), dpi=120, facecolor='w', edgecolor='k')
    plt.plot(moving_aves, color='b',label="Numerical Simulation",linewidth=2)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    plt.ylabel("Reward",fontsize=30)
    plt.xlabel("Episode",fontsize=30)
    plt.legend(loc="lower right", fontsize=30)
    plt.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.7)
    plt.title('Reward Per Episode',fontsize=30)
    plt.savefig(path + "/model_inverted_numerical/reward")
    plt.close()


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    dqn = DQN_Agent(env)
    dqn.start()
