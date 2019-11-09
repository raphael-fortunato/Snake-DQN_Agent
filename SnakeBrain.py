import random
import pygame
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, Flatten
import Snake
from collections import deque
import time as tm
import CustomTensorBoard
from statistics import mean
from math import log
from os import listdir
from os.path import isfile, join



class DQNAgent(object):

    def __init__(self, epsilon_decay, model = None,epsilon=1.0, epsilon_min=.1, gamma =0.99):
        self.game = Snake.Game(GRID_SIZE, True)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma             
        self.model = self.build_model() if model == None else model
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.memory1 = deque(maxlen=500000)
        self.memory2 = deque(maxlen = 500000)
        self.tensorboard = CustomTensorBoard.ModifiedTensorBoard(log_dir = f"logs")
        self.eta = .8
        self.eta_min = .5

    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(16, (8,8), strides=4, padding='same', input_shape=(60,60,1,)))
        model.add(Activation('relu'))

        model.add(Convolution2D(32, (4,4), strides=2, padding = 'same'))
        model.add(Activation('relu'))

        model.add(Convolution2D(32, (3,3), strides=1, padding = 'same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))

        model.add(Dense(4))
        model.add(Activation('linear'))
        adam= Adam(lr=0.001)
        model.compile(loss='mean_squared_error',
                        optimizer=adam)
        return model

    def action(self, state):
        adapt_ep = self.epsilon * (1 + log(len(self.game.snake.body) - 2))
        if random.random() <= min(1, adapt_ep) and TRAINING:
            return random.randint(0,3)
        else: 
            return np.argmax(self.target_model.predict(state)[0])

    def penelize_memory(self, penelize ,deque_list):
        deque_list =np.array(deque_list)
        if penelize:
            for i, c in enumerate(deque_list):
                deque_list[i][3] -= .1
        self.memory2 += deque(deque_list)

    def remember(self, important ,state, action, reward, next_state, done): 
        if important:     
            self.memory1.append((state, action, reward, next_state, done))
        else:
            self.memory2.append((state, action, reward, next_state, done))


    def Train(self, episode): 
        if len(self.memory1) > BATCH_SIZE * 5:
            M1_batch = random.sample(self.memory1, int(min(BATCH_SIZE, len(self.memory1)) *  self.eta))
            M2_batch = random.sample(self.memory2, int(min(BATCH_SIZE, len(self.memory1)) * (1-self.eta)))
            minibatch =  np.concatenate((M1_batch, M2_batch), axis = 0)

            s_batch = [data[0] for data in minibatch]
            a_batch = [data[1] for data in minibatch]
            r_batch = [data[2] for data in minibatch]
            st1_batch = [data[3] for data in minibatch]
            d_batch = [data[4] for data in minibatch]
            q_batch = []
            target_f = []
            for j in range(len(minibatch)):
                q_batch.append( r_batch[j] + (self.gamma * (1 - d_batch[j]) *np.amax(self.target_model.predict(st1_batch[j]))) )
                target_f.append(self.model.predict(s_batch[j]))
                target_f[j] [0][a_batch[j]] = q_batch[j]

            s_batch = np.array(s_batch).reshape(len(minibatch), INPUT_SIZE[0], INPUT_SIZE[1], 1)
            target_f = np.array(target_f).reshape(len(minibatch),4)
            self.model.fit(s_batch, target_f,batch_size = BATCH_SIZE epochs = 1, verbose = 0, callbacks = [self.tensorboard])

        if(self.epsilon > self.epsilon_min):
            self.epsilon = -(episode / EPISODE) + 1.
        if(self.eta > self.eta_min):
            self.eta = -.4 * (episode / EPISODE) + .8
        if  episode % UPDATE_TARGET == 0:
            print("***update target model***")
            self.target_model.set_weights(self.model.get_weights())


    


TRAINING = False
BATCH_SIZE = 2_000
EPISODE =1_000_000
UPDATE_TARGET = 1_000
GRID_SIZE = 12
INPUT_SIZE = (60,60, )
AGGREGATE_STATS_EVERY = 100
mypath = "/home/rfortunato1994/logs"

if __name__ == '__main__':
    m = load_model("gcp\\episode-1000000")
    #m = None
    agent = DQNAgent(0.999, model = m, epsilon = 1.)
    action_size = 4
    done = False
    count= 0
    total_frames = 0
    total_exp = 0
    ep_rewards = []
    average_step = 0
    temp_memory = []
    total_time = []
    test_run = []
    test_run.append(0)
    for episode in range(EPISODE + 1):
        agent.tensorboard.step = episode
        episode_reward = 0
        agent.game.snake.reset(GRID_SIZE //2,GRID_SIZE//2)
        agent.game.resetfood()
        state = agent.game.GenerateImage()
        state /= 255
        state = np.reshape(state, (1, 60,60,1))
        timestep = 100
        skip_memory = 0
        time = 0
        while timestep >= 0:
            time += 1
            action = agent.action(state)
            next_state, reward, done = agent.game.nextstate(action)
            next_state /=  255
            next_state = np.reshape(next_state, (1, 60,60,1))
            if abs(reward) >= .5 and skip_memory == 0:
                agent.remember(True, state,action, reward, next_state, done)
                agent.penelize_memory(False, temp_memory)
                temp_memory.clear()
                timestep = int(2 * len(agent.game.snake.body) + 100)
                skip_memory =2
            elif skip_memory == 0:
                temp_memory.append((state,action, reward, next_state, done))
            state = next_state
            count += 1
            episode_reward += reward
            timestep -= 1
            skip_memory = skip_memory -1 if skip_memory > 0 else 0
            if done or timestep == 0:
                # if not TRAINING:
                #     TRAINING = True
                #     test_run.append((episode_reward,len(agent.game.snake.body)-3))
                total_time.append(time)
                time = 0
                if timestep == 0:
                    #average_step = (average_step * episode + time)//(episode +1)
                    print("bad memories")
                    agent.penelize_memory(True, temp_memory)
                    temp_memory.clear()
                print(f"episode: {episode}/{EPISODE},  epsilon: {agent.epsilon}, eta: {agent.eta}"
                )
                ep_rewards.append(episode_reward  )
                # if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                #     average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                #     min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                #     max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                #     average_time  = mean(total_time[-AGGREGATE_STATS_EVERY:])
                #     agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,average_episode_length=average_time ,epsilon=agent.epsilon, test_run_reward=test_run[-1][0], test_run_score =test_run[-1][0] )
                    # onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
                    # print(onlyfiles)
                break
        try:
            if(((episode % 50_000 == 0 and episode != 0)or episode == 1) and TRAINING):
                model_save_name = f"episode{episode}"
                agent.model.save(model_save_name)
        except:
            pass
        if  episode % 100 == 0 and episode != 0 and TRAINING:
            total_exp += min(BATCH_SIZE, len(agent.memory1))
            print(f"***Training*** \n ***memory size: {len(agent.memory1)}, {len(agent.memory2)}*** \
            \n Total experiences replayed : {total_exp}")
            agent.Train(episode)
            TRAINING = False







