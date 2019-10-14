import random
import pygame
import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Convolution2D, Flatten, Dropout, Input, concatenate 
import Snake
from collections import deque
import time as tm
import CustomTensorBoard
import cv2
from matplotlib import pyplot

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)


class DQNAgent(object):

    def __init__(self, epsilon_decay, model = None,epsilon=1.0, epsilon_min=.2, gamma =0.99):
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
        self.eta = .8
        self.eta_min = .5

    def build_model(self):
        input1 = Input(shape=(INPUT_SIZE,  ))

        feature_model = Dense(12, activation='relu') (input1)
        feature_model = Dropout(0.2) (feature_model) 

        feature_model = Dense(36, activation='relu') (feature_model)
        feature_model = Dropout(0.2) (feature_model) 


        input2 = Input(shape=(144, ))
        
        array_model = Dense(512, activation='relu') (input2)
        array_model = Dropout(0.2) (array_model)

        array_model = Dense(1024, activation='relu') (array_model)
        array_model = Dropout(0.2) (array_model)

        combined_model = concatenate([feature_model,array_model], axis = 1)

        combined_model = Dense(512, activation='relu') (combined_model)
        combined_model = Dropout(0.2) (combined_model)

        combined_model = Dense(4, activation='linear') (combined_model)
        adam= Adam(lr=0.001)
        combined_model = Model([input1, input2], combined_model)
        combined_model.compile(loss='mean_squared_error',
                        optimizer=adam)
        return combined_model

    def action(self, state):
        #epsilon_step = self.epsilon * (1 + (step / (average+ 1)))
        if random.random() <= self.epsilon and TRAINING:
            return random.randint(0,3)
        else: 
            return np.argmax(self.target_model.predict(state)[0])

    def remember(self, important ,feature_state, array_state, action, reward, feature_next_state, array_next_state, done): 
        if important:     
            self.memory1.append((feature_state, array_state, action, reward, feature_next_state, array_next_state, done))
        else:
            self.memory2.append((feature_state, array_state, action, reward, feature_next_state, array_next_state, done))

    def penelize_memory(self, penelize ,deque_list):
        deque_list =np.array(deque_list)
        if penelize:
            for i, c in enumerate(deque_list):
                deque_list[i][3] -= .1
        self.memory2 += deque(deque_list)


    def Train(self, episode): 
        M1_batch = random.sample(self.memory1, int(min(BATCH_SIZE, len(self.memory1)) *  self.eta))
        M2_batch = random.sample(self.memory2, int(min(BATCH_SIZE, len(self.memory1)) * (1-self.eta)))
        minibatch =  np.concatenate((M1_batch, M2_batch), axis = 0)

        feature_s_batch = [data[0] for data in minibatch]
        array_s_batch = [data[1] for data in minibatch]
        a_batch = [data[2] for data in minibatch]
        r_batch = [data[3] for data in minibatch]
        feature_st1_batch = [data[4] for data in minibatch]
        array_st1_batch = [data[5] for data in minibatch]
        d_batch = [data[6] for data in minibatch]
        q_batch = []
        target_f = []
        #print(np.array(st1_batch[0]).shape)
        for j in range(len(minibatch)):
            q_batch.append( r_batch[j] + (self.gamma * (1 - d_batch[j]) *np.amax(self.target_model.predict([feature_st1_batch[j], array_st1_batch[j]])))) 
            target_f.append(self.model.predict([feature_s_batch[j], array_s_batch[j]]))
            target_f[j] [0][a_batch[j]] = q_batch[j]

        feature_s_batch, array_s_batch = np.array(feature_s_batch).reshape(len(minibatch), INPUT_SIZE), np.array(array_s_batch).reshape(len(minibatch), 144)
        target_f = np.array(target_f).reshape(len(minibatch),4)
        self.model.fit([feature_s_batch, array_s_batch], target_f, epochs = 1, batch_size = len(minibatch), verbose = 0, callbacks = [self.tensorboard])

        if(self.epsilon > self.epsilon_min):
            self.epsilon = -(episode / EPISODE) + .5
        if(self.eta > self.eta_min):
            self.eta = -.4 * (episode / EPISODE) + .8
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

def DrawFeatures(window, grid, size, features, timestep):
    pygame.font.init() 
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    for i, feature in enumerate(features):
        text = myfont.render(str(feature), False, (250,250,250))
        window.blit(text,(20,10 + i * 30))
    text = myfont.render(str(timestep), False,(250,250,250))
    window.blit(text,(20,160))

def Draw(window, clock, Grid, feature, timestep):
    size = 40 * GRID_SIZE
    row = GRID_SIZE
    KeyEvent()
    pygame.time.delay(50)
    clock.tick(10)
    DrawWindow(window, size, row)
    DrawObjects(window,Grid,size, row)
    DrawFeatures(window, Grid, size, feature, timestep)
    pygame.display.update()


MODEL_NAME = "double network 2nd"
TRAINING = False
BATCH_SIZE = 2000
EPISODE =100_001
UPDATE_TARGET = 5_000
SCREEN = True
GRID_SIZE = 12
INPUT_SIZE = 5
AGGREGATE_STATS_EVERY = 100


if __name__ == '__main__':
    m = load_model("C:\\Users\\Raphael Fortunato\\Documents\\Python\\Snake-DQN_Agent\\Colab\\episode200_000")
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
    temp_memory = []
    for episode in range(EPISODE + 1):
        agent.tensorboard.step = episode
        episode_reward = 0
        agent.game.snake.reset(GRID_SIZE //2,GRID_SIZE//2)
        agent.game.resetfood()
        feature_state, array_state = agent.game.get_features() , agent.game.generateGrid()
        if SCREEN:
            Draw(window, clock, array_state, feature_state, 0)
        feature_state, array_state = feature_state.reshape(1,INPUT_SIZE), array_state.reshape(1,144)
        timestep = 30
        skip_memory = 0
        while timestep >= 0:
            action = agent.action([feature_state, array_state])
            feature_next_state, array_next_state, reward, done = agent.game.nextstate(action)
            if SCREEN:
                Draw(window, clock, agent.game.generateGrid(), feature_next_state, timestep)
            feature_next_state, array_next_state = feature_next_state.reshape(1, INPUT_SIZE), array_next_state.reshape(1,144)
            if abs(reward) >= .5 and skip_memory == 0:
                agent.remember(True, feature_state, array_state, action, reward, feature_next_state, array_next_state, done)
                agent.penelize_memory(False, temp_memory)
                temp_memory.clear()
                timestep = int(2 * len(agent.game.snake.body) + 30)
                skip_memory =2
            elif skip_memory == 0:
                temp_memory.append((feature_state, array_state, action, reward, feature_next_state, array_next_state, done))
            feature_state, array_state = feature_next_state, array_next_state
            count += 1
            episode_reward += reward
            timestep -= 1
            skip_memory = skip_memory -1 if skip_memory > 0 else 0
            if done or timestep == 0:
                if timestep == 0:
                    #average_step = (average_step * episode + time)//(episode +1)
                    print("bad memories")
                    agent.penelize_memory(True, temp_memory)
                    temp_memory.clear()
                #average_step = (average_step * episode + time)//(episode +1)
                print(f"episode: {episode}/{EPISODE},  epsilon: {agent.epsilon}, eta: {agent.eta}"
                )
                ep_rewards.append(episode_reward)
                
                if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    
                    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward ,epsilon=agent.epsilon)
                break
        if  episode % 100 == 0 and episode != 0 and TRAINING:
            total_exp += min(BATCH_SIZE, len(agent.memory1))
            print(f"***Training*** \n ***memory size: {len(agent.memory1)}, {len(agent.memory2)}*** \
            \n Total experiences replayed : {total_exp}")
            agent.Train(episode)
        try:
            if((episode % 50000 == 0 and episode != 0) and TRAINING):
                agent.model.save(f"Snake_model_{episode + 1_000_000}_double_NN-12x12G" )
        except:
            pass






