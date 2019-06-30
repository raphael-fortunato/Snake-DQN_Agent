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


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)


class DQNAgent(object):

    def __init__(self, epsilon_decay, model = None,epsilon=1.0, epsilon_min=.1, gamma =0.99):
        self.game = Snake.Game(GRID_SIZE)
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
        model.add(Dense(512, input_shape=(INPUT_SIZE, )))
        model.add(Activation('relu'))

        model.add(Dense(1024))
        model.add(Activation('relu'))

        model.add(Dense(256))
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

        s_batch = np.array(s_batch).reshape(len(minibatch), INPUT_SIZE)
        target_f = np.array(target_f).reshape(len(minibatch),4)
        self.model.fit(s_batch, target_f, epochs = 1, batch_size = len(minibatch), verbose = 0, callbacks = [self.tensorboard])

        if(self.epsilon > self.epsilon_min):
            self.epsilon = -(episode / EPISODE) + 1.
        if(self.eta > self.eta_min):
            self.eta = -.4 * (episode / EPISODE) + .6
        if  episode % UPDATE_TARGET == 0:
            print("***update target model***")
            self.target_model.set_weights(self.model.get_weights())


def DrawObjects(window, grid, size, row):
    distance = size // row
    for x in range(row):
        for y in range(row):
            if grid[x][y] == 1:
                pygame.draw.rect(window, (0,255,0), (x * distance, y * distance, distance, distance))
            elif grid[x][y] == 2:
                pygame.draw.rect(window, (0,255,0), (x * distance, y * distance, distance, distance))
            elif  grid[x][y] == 3:
                pygame.draw.rect(window, (255,0,0), (x * distance, y * distance, distance, distance))

def DrawGrid(window, size, rows):
    sizebetween = size // rows
    x = 0
    y = 0
    for l in range(rows):
        x = x + sizebetween
        y = y + sizebetween
        pygame.draw.line(window, (255, 255, 255), (x, 0), (x, size))
        pygame.draw.line(window, (255, 255, 255), (0, y), (size, y))

def DrawWindow(window, size, rows):
    window.fill((0,0,0))
    DrawGrid(window, size, rows)
    pygame.display.update()

def KeyEvent():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            keys = pygame.key.get_pressed()
            for key in keys:
                if keys[pygame.K_s]:
                    agent.model.save(f"Snake_model_manualsave_{tm.time()}")
                    break


def Draw(window, clock, Grid):
    size = 40 * GRID_SIZE
    row = GRID_SIZE
    KeyEvent()
    pygame.time.delay(10)
    clock.tick(10)
    DrawWindow(window, size, row)
    DrawObjects(window,Grid,size, row)
    pygame.display.update()


MODEL_NAME = "Dense-4l-512-1024-256-4-12x12G - sixth"
TRAINING = False
BATCH_SIZE = 5_000
EPISODE =150_000
UPDATE_TARGET = 10_000
SCREEN = True
GRID_SIZE = 12
INPUT_SIZE = GRID_SIZE * GRID_SIZE
AGGREGATE_STATS_EVERY = 100


if __name__ == '__main__':
    m = load_model("C:\\Users\\Raphael Fortunato\\Documents\\Python\\Snake-DQN_Agent\\Snake_model_500000_dense-12x12G")
    #m = None
    agent = DQNAgent(0.999, model = m, epsilon = 1.)
    if SCREEN:
        size = 40 * GRID_SIZE
        pygame.init()
        window = pygame.display.set_mode((size, size))
        clock = pygame.time.Clock()
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
        state = agent.game.generateGrid()
        if SCREEN:
            Draw(window, clock, state)
        state = state.reshape(1,INPUT_SIZE) 
        for time in range(500):
            action = agent.action(state, time, average_step)
            next_state, reward, done = agent.game.nextstate(action)
            if SCREEN:
                Draw(window, clock, next_state)
            next_state = next_state.reshape(1, INPUT_SIZE)
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
                agent.model.save(f"Snake_model_{episode+ 390_000}_dense-12x12G" )
        except:
            pass






