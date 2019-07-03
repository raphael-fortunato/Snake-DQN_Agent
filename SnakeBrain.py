import random
import pygame
import tkinter as tk
from tkinter import messagebox
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
import cv2


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


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
        self.tensorboard = CustomTensorBoard.ModifiedTensorBoard(log_dir = f"logs\\{MODEL_NAME}-{int(tm.time())}")
        self.eta = .6
        self.eta_min = .5

    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, (8,8), strides=4, padding='same', input_shape=(84,84,1)))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, (4,4), strides=2, padding = 'same'))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, (3,3), strides=1, padding = 'same'))
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

    def action(self, state, step, average):
        epsilon_step = self.epsilon * (1 + (step / (average+ 1)))
        if random.random() <= min(epsilon_step, 1) and TRAINING:
            return random.randint(0,3)
        else: 
            return np.argmax(self.target_model.predict(state)[0])

    def remember(self, important ,state, action, reward, next_state, done): 
        if important:     
            self.memory1.append((state, action, reward, next_state, done))
        else:
            self.memory2.append((state, action, reward, next_state, done))


    def Train(self, episode): 
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
        self.model.fit(s_batch, target_f, epochs = 1, verbose = 0, callbacks = [self.tensorboard])

        if(self.epsilon > self.epsilon_min):
            self.epsilon = -(episode / EPISODE) + 1.
        if(self.eta > self.eta_min):
            self.eta = -.4 * (episode / EPISODE) + .6
        if  episode % UPDATE_TARGET == 0:
            print("***update target model***")
            self.target_model.set_weights(self.model.get_weights())

def Preprocess(image):
    assert image.ndim == 3
    state = image
    state = cv2.cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state, (84,84) )
    state = np.array(state).reshape([-1, state.shape[0], state.shape[1], 1])
    state = state.astype(float)
    state /= 255
    return state 

    

MODEL_NAME = "Conv2D-4l-32-64-64-512-4-12x12G - test"
TRAINING = True
BATCH_SIZE = 5_000
EPISODE =10_000
UPDATE_TARGET = 1_000
GRID_SIZE = 12
INPUT_SIZE = (84,84, )
AGGREGATE_STATS_EVERY = 100


if __name__ == '__main__':
    #m = load_model("C:\\Users\\Raphael Fortunato\\Documents\\Python\\Snake-DQN_Agent\\Snake_model_500000_dense-12x12G")
    m = None
    agent = DQNAgent(0.999, model = m, epsilon = 1.)
    action_size = 4
    done = False
    count= 0
    total_frames = 0
    total_exp = 0
    ep_rewards = []
    average_step = 0
    for episode in range(EPISODE + 1):
        agent.tensorboard.step = episode
        episode_reward = 0
        agent.game.snake.reset(GRID_SIZE //2,GRID_SIZE//2)
        agent.game.resetfood()
        state = agent.game.Init()
        state = Preprocess(state)
        for time in range(500):
            action = agent.action(state, time, average_step)
            next_state, reward, done = agent.game.nextstate(action)
            next_state = Preprocess(next_state)
            if abs(reward) >= .5:
                agent.remember(True, state,action, reward, next_state, done)
            else:
                agent.remember(False, state, action, reward, next_state, done)
            state = next_state
            count += 1
            episode_reward += reward
            if done:
                average_step = (average_step * episode + time)//(episode +1)
                total_frames += time
                print(f"episode: {episode}/{EPISODE}, total frames: {total_frames},  epsilon: {agent.epsilon}, eta: {agent.eta}"
                )
                ep_rewards.append(episode_reward)
                
                if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,average_episode_length=time ,epsilon=agent.epsilon)
                break
            if time == 499:
                average_step = (average_step * episode + time)//(episode +1)
        if  episode % 100 == 0 and episode != 0 and TRAINING:
            total_exp += min(BATCH_SIZE, len(agent.memory1))
            print(f"***Training*** \n ***memory size: {len(agent.memory1)}, {len(agent.memory2)}*** \
            \n Total experiences replayed : {total_exp}")
            agent.Train(episode)
        try:
            if((episode % 5000 == 0 and episode != 0)):
                agent.model.save(f"Snake_model_{episode+ 390_000}_Conv2D-12x12G" )
        except:
            pass






