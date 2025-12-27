import pygame
from collections import defaultdict, deque
import numpy as np
from json import dump,load
import random as rd
from keras.layers import Dense,Input
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from tqdm import tqdm
#import DQN_class_
import requests


tick = 400
map_size = 10
class Snake:
    def __init__(self):
        self.map_size = (map_size,map_size)#(60,40)
        self.head = [self.map_size[0]//2,self.map_size[1]//2]
        self.tail = [[self.map_size[0]//2+1,self.map_size[1]//2]]
        self.heading = 0 # 좌 상 우 하
        self.apple = [self.map_size[0]//3,self.map_size[1]//3]
        self.score = 0
    def step(self,ro): # -1좌 0직 1우
        self.heading = (self.heading + ro)%4
        self.tail.append(list(self.head))
        score_add = 0
        if not(self.apple == self.head):
            self.tail.pop(0)
        else:
            while(True):
                sub = [rd.randint(0,self.map_size[0]),rd.randint(0,self.map_size[1])]
                if not(sub in ([self.head] + self.tail) ):
                    self.apple = sub
                    break
            self.score += 1
            score_add = 1


        if(self.heading == 0):
            self.head[0] -= 1
        elif (self.heading == 1):
            self.head[1] -= 1
        elif(self.heading == 2):
            self.head[0] += 1
        elif (self.heading == 3):
            self.head[1] += 1

        if (self.head[0]<0 or self.head[0] > self.map_size[0]) or\
                (self.head[1]<0 or self.head[1] > self.map_size[1]) or \
                (self.head in self.tail):
            #print("Game_over : ",self.score)
            score_add = -1

        return score_add

def get_state1(snake):
    board = np.zeros(snake.map_size[0]*snake.map_size[1]).reshape(snake.map_size)
    board[snake.head[0]][snake.head[1]] = snake.heading+1
    for t in snake.tail:
        board[t[0]][t[1]] = 5
    if(snake.heading==0)and(snake.apple[0]<=snake.head[0]): board[snake.apple[0]][snake.apple[1]] = -1
    elif(snake.heading==1)and(snake.apple[1]<=snake.head[1]): board[snake.apple[0]][snake.apple[1]] = -1
    elif (snake.heading == 2) and (snake.apple[0] >= snake.head[0]): board[snake.apple[0]][snake.apple[1]] = -1
    elif (snake.heading == 1) and (snake.apple[1] >= snake.head[1]): board[snake.apple[0]][snake.apple[1]] = -1

    return board
s = Snake()
while(True):
    s.step(int(input()))
    print(get_state1(s))