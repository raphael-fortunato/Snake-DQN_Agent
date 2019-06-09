import random
import pygame
pygame.init()
import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten
from math import sqrt, cos, degrees


class cube(object):
    global x,y
    rows = 20
    w = 800


    def __init__(self, x, y, color=(0,255,0)):
        self.x = x
        self.y = y
        self.color = color

    def move(self, dirnx, dirny):
        pass

    def Draw(self, window):
        distance = self.w // self.rows
        pygame.draw.rect(window, self.color, (self.x * distance, self.y * distance, distance, distance))


class Snake(object):
    global body,dx, dy, tail
    def __init__(self, color, x, y):
        self.color = color
        self.body = []
        self.InitSnake(3)
        self.tail = cube(self.body[-1].x,self.body[-1].y +1)
        self.dx = 0
        self.dy = -1

    def MoveRight(self):
        if self.dx == -1:
            return
        self.dy = 0
        self.dx = 1

    def MoveLeft(self):
        if self.dx == 1:
            return
        self.dy = 0
        self.dx = -1

    def MoveDown(self):
        if self.dy == -1:
            return
        self.dy = 1
        self.dx = 0

    def MoveUp(self):
        if self.dy == 1:
            return
        self.dy = -1
        self.dx = 0

    def InitSnake(self, start_size):
        for i in range(start_size):
            self.body.append(cube(10, 10 +i))


    def reset(self, x, y):
        self.body.clear()
        self.InitSnake(3)
        self.tail = cube(x, y + 1)
        self.dx = 0
        self.dy = -1

    def addCube(self):
        self.body.append(cube(self.tail.x, self.tail.y, color = (0,255,0)))

    def Draw(self, window):
        for i in self.body:
            i.Draw(window)


    def MoveSnake(self):
        self.tail.x = self.body[-1].x
        self.tail.y = self.body[-1].y
        for i in range(len(self.body) -1, 0, -1):
            self.body[i].x = self.body[i - 1].x
            self.body[i].y = self.body[i - 1].y
        self.body[0].y += self.dy
        self.body[0].x += self.dx
        # for i,block in enumerate(self.body):
        #     print(f"block {i} location: {block.x} , {block.y}")

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


def randomSnack(window, rows, item):
    position = item.body
    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter( lambda z: z.x ==  x and z.y == y, position))):
            continue
        else:
            break

    food = cube(x,y, color = (255,0,0))
    return food


def message_box(subject, content):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    messagebox.showinfo(subject, content)
    try:
        root.destroy()
    except:
        pass

def IsWithin(value, min, max):
    return value >= min and value <= max


class Game(object):

    def __init__(self, training):
        global snake, food, rows, Training, window
        if training: 
            self.Training = training
            self.size = 800
            self.rows = 20
            self.window = None
            self.snake = Snake((0,255,0), 10,10)
            self.food = randomSnack(self.window,self.rows, self.snake)

    def resetfood(self):
        self.food = randomSnack(self.window, self.rows, self.snake)

    def generateGrid(self):
        grid = np.zeros((20,20))
        for c,i in reversed(list(enumerate(self.snake.body))):
            try:
                if c == 0:
                    #2 is head
                    if i.x == -1 or i.y == -1:
                        pass
                    else:
                        grid[i.x][i.y] = 2
                else:
                    #1 is body
                    grid[i.x][i.y] = 1
            except:
                grid[self.food.x][self.food.y] = 3
                return grid
            #is food
        grid[self.food.x][self.food.y] = 3
        return grid

    def CalcDistance(self,a1,a2, b1, b2):
        x_dist = (a1 - a2)**2
        y_dist = (b1 -b2)**2
        return sqrt(x_dist+ y_dist)

    def MapValue(self, curmin,  curmax, tarmin,  tarmax, curval):
        return tarmin + (tarmax - tarmin) * ((curval - curmin)/ (curmax - curmin))

    def CalcAngle(self, dir, distC):
        max_dist = self.CalcDistance(20,0,20,0)
        if distC == max_dist:
            return max_dist
        elif distC == 0:
            return 0

        if dir[1] != 0:
            distA = self.CalcDistance(self.snake.body[0].x,self.snake.body[0].x , self.snake.body[0].y, self.snake.body[0].x) 
        elif dir[0] != 0:
            distA = self.CalcDistance(self.snake.body[0].x, self.food.x, self.snake.body[0].y, self.snake.body[0].y)
        angle = cos(distA/ distC)
        degree = degrees(angle)
        return self.MapValue(0, max_dist/2, 0, max_dist,((90 - degree) / 90)* distC)
        
                

    def CalcDistancePos(self, new, old, food):
        oldxdist = (food[0] - old[0]) * (food[0] - old[0])
        oldydist = (food[1] - old[1]) * (food[1] - old[1])
        old_dist = sqrt(oldxdist + oldydist)

        newxdist = (food[0] - new[0]) * (food[0] - new[0])
        newydist = (food[1] - new[1]) * (food[1] - new[1])
        new_dist = sqrt(newxdist + newydist)

        return 1 if old_dist > new_dist else -.5

    def CalcFeatures(self):
        max_dist = self.CalcDistance(20,0,20,0)
        wall_up = self.CalcDistance(self.snake.body[0].x, self.snake.body[0].x, self.snake.body[0].y, -1)
        wall_down = self.CalcDistance(self.snake.body[0].x, self.snake.body[0].x, self.snake.body[0].y, 20)
        wall_left = self.CalcDistance(self.snake.body[0].x, -1, self.snake.body[0].y, self.snake.body[0].y)
        wall_right = self.CalcDistance(self.snake.body[0].x, 20, self.snake.body[0].y, self.snake.body[0].y)

        food_up = max_dist if self.snake.body[0].y < self.food.y else self.CalcDistance(self.snake.body[0].x, self.food.x, self.snake.body[0].y, self.food.y)
        food_down = max_dist if self.snake.body[0].y > self.food.y else self.CalcDistance(self.snake.body[0].x, self.food.x, self.snake.body[0].y, self.food.y)
        food_left = max_dist if self.snake.body[0].x < self.food.x else self.CalcDistance(self.snake.body[0].x, self.food.x, self.snake.body[0].y, self.food.y)
        food_right = max_dist if self.snake.body[0].x > self.food.x else self.CalcDistance(self.snake.body[0].x, self.food.x, self.snake.body[0].y, self.food.y)
        #print(food_up, food_down, food_left, food_right)
        food_up = self.CalcAngle((0,1), food_up)
        food_down = self.CalcAngle((0,-1), food_down)
        food_left = self.CalcAngle((1,0), food_left)
        food_right = self.CalcAngle((-1,0), food_right)
        #print(food_up, food_down, food_left, food_right)
        body_up = max_dist if len(list(filter(lambda x: x.y < self.snake.body[0].y and x.x == self.snake.body[0].x, self.snake.body))) == 0 \
                    else self.CalcDistance(self.snake.body[0].x, self.snake.body[0].x, self.snake.body[0].y,list(filter(lambda x: x.y < self.snake.body[0].y, self.snake.body))[0].y )
        body_down = max_dist if len(list(filter(lambda x: x.y > self.snake.body[0].y and x.x == self.snake.body[0].x, self.snake.body))) == 0 \
                    else self.CalcDistance(self.snake.body[0].x, self.snake.body[0].x, self.snake.body[0].y,list(filter(lambda x: x.y > self.snake.body[0].y, self.snake.body))[0].y )
        body_left = max_dist if len(list(filter(lambda x: x.x < self.snake.body[0].x and x.y == self.snake.body[0].y, self.snake.body))) == 0 \
                    else self.CalcDistance(self.snake.body[0].x, list(filter(lambda x: x.x < self.snake.body[0].x, self.snake.body))[0].x, self.snake.body[0].y, self.snake.body[0].y )
        body_right = max_dist if len(list(filter(lambda x: x.x > self.snake.body[0].x and x.y == self.snake.body[0].y, self.snake.body))) == 0 \
                    else self.CalcDistance(self.snake.body[0].x, list(filter(lambda x: x.x > self.snake.body[0].x, self.snake.body))[0].x, self.snake.body[0].y, self.snake.body[0].y )
        features =  [wall_up, wall_down, wall_left, wall_right, food_up, food_down, food_left, food_right, body_up, body_down, body_left, body_right]
        features = list(map(lambda x:self.MapValue(0,max_dist, 1, 0, x), features))
        print(features)
        features = np.array(features)
        return features.reshape(12, 1)

    def checkcollision(self, new_pos, old_pos, food): 
        for i in range(1, len(self.snake.body),1):
            if(self.snake.body[0].x == self.snake.body[i].x and self.snake.body[0].y == self.snake.body[i].y):
                    print(f"Score: {len(self.snake.body) -3}")
                    #message_box("You lost!", "Play again..")
                    self.snake.reset(10,10)
                    return -20, True
        if(not IsWithin(self.snake.body[0].x, 0, self.rows -1) or not IsWithin(self.snake.body[0].y, 0, self.rows -1)):
            print(f"Score: {len(self.snake.body) -3}")
            #message_box("You lost!", "Play again..")
            self.snake.reset(10, 10)
            return -20, True
        elif(self.snake.body[0].x == self.food.x and self.snake.body[0].y == self.food.y):
            self.snake.addCube()
            self.food = randomSnack(self.window,self.rows,self.snake)
            return 20 , False
        distance = self.CalcDistancePos(new_pos,old_pos,food)
        return distance , False


    def performrandomaction(self, rand):
            if(rand ==0):
                self.snake.MoveDown()
            elif(rand == 1):
                self.snake.MoveUp()  
            elif(rand == 2):
                self.snake.MoveRight()
            elif(rand == 3):  
                self.snake.MoveLeft()


    def nextstate(self, action_index):
        if self.Training:
            old_pos_head  = (self.snake.body[0].x, self.snake.body[0].y)
            pos_f = (self.food.x, self.food.y)
            self.performrandomaction(action_index)
            self.snake.MoveSnake()
            features = self.CalcFeatures()
            new_pos_head  = (self.snake.body[0].x, self.snake.body[0].y)
            reward, done = self.checkcollision(new_pos_head, old_pos_head, pos_f)
            
            return features, reward, done

