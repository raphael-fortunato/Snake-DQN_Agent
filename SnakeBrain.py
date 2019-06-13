import random
import pygame
import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout, CuDNNLSTM, TimeDistributed
import Snake
from collections import deque
import time as tm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)





class DQNAgent(object):

    def __init__(self, epsilon_decay, model = None,epsilon=1.0, epsilon_min=.1, gamma =0.9):
        self.game = Snake.Game(True)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma             
        self.model = self.build_model() if model == None else model
        self.memory = deque(maxlen=10000)

    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(16, (3, 3),input_shape=(20,20,1) ))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, (3, 3)))
        model.add(Activation('relu'))
        
        # model.add(TimeDistributed(Flatten()))

        # model.add(CuDNNLSTM(128, return_sequences = True))
        # model.add(Dropout(.2))
        # model.add(Activation("relu"))

        # model.add(CuDNNLSTM(128,  return_sequences = False))
        # model.add(Dropout(.2))
        # model.add(Activation("relu"))

        # model.add(Dense(1048))
        # model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))

        model.add(Dense(4))
        model.add(Activation('linear'))
        adam= Adam(lr=0.001)
        model.compile(loss='mean_squared_error',
                        optimizer=adam)
        print(model.summary())
        return model

    def action(self, state):
        if random.random() <= self.epsilon and TRAINING:
            return random.randint(0,3)
        else: 
            return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, next_state, done):      
        self.memory.append((state, action, reward, next_state, done))

    def Train(self, episode):
        if(agent.epsilon > agent.epsilon_min):
            agent.epsilon = -1.5 * (episode / EPISODE) + self.epsilon
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * 
                            np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

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
    size = 800
    row = 20
    KeyEvent()
    pygame.time.delay(0)
    clock.tick(10)
    DrawWindow(window, size, row)
    DrawObjects(window,Grid,size, row)
    pygame.display.update()

TRAINING = True
BATCH_SIZE = 8
EPISODE =5000
SCREEN = True

if __name__ == '__main__':
    while True:
        m = load_model("Snake_model_manualsave_Furtest_trained")
        agent = DQNAgent(0.999, model = m, epsilon = 0.)
        if SCREEN:
            size = 800
            pygame.init()
            window = pygame.display.set_mode((size, size))
            clock = pygame.time.Clock()
        action_size = 4
        done = False

        for episode in range(EPISODE):
            agent.game.resetfood()
            state = agent.game.generateGrid()
            if SCREEN:
                Draw(window, clock, state)
            state = state.reshape(-1,20,20,1)
            for time in range(500):
                action = agent.action(state)
                next_state, reward, done = agent.game.nextstate(action)
                if SCREEN:
                    Draw(window, clock, next_state)
                next_state = next_state.reshape(-1,20,20,1)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {},  e: {:.2}"
                        .format(episode, EPISODE, time,  agent.epsilon))
                    break
                if len(agent.memory) > BATCH_SIZE:
                    agent.Train(episode)
            if(episode % 200 == 0 or episode == 4999):
                agent.model.save(f"Snake_model_{episode}" )






