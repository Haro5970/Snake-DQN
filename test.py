import matplotlib.pyplot as plt
import pygame
import random as rd
import numpy as np
import requests
from json import load, dump, dumps

tick = 400
map_size = 20
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


headers = {"Content-Type": "application/json"}
class DQLearner:
    drawing_data = {}
    drawing_data['block_size'] = 20
    drawing_data['SCREEN'] = (Snake().map_size[0] * drawing_data['block_size'],
                              Snake().map_size[1] * drawing_data['block_size'])
    drawing_data['color'] = {'BG': (150, 255, 150), "line": (200, 200, 200), 'sight': (200,255,200)}
    file_direct = ''


    def __init__(self, name):
        self.name = name
        self.ver = 'D0'
        try:
            with open(self.file_direct+name + "_obj.json", 'r', encoding='UTF-8') as f:
                data = load(f)
            self.rewards = data['rewards']
            print("Load data!")
        except:
            self.rewards = [0]

    def act(self, state):
        action = requests.post("http://127.0.0.1:8000/"+self.name+"/act",headers=headers,data=dumps({'state':state})).text
        return int(action)

    def memorize(self, state, action, reward, next_state):
        dic = {'state':state,
               'action':action,
               'reward':reward,
               'next_state':next_state}
        if(reward>0):
            for i in range(5):
                requests.put("http://127.0.0.1:8000/"+self.name+"/memorize",headers=headers,data=dumps(dic))
        else:
            requests.put("http://127.0.0.1:8000/"+self.name+"/memorize",headers=headers,data=dumps(dic))



    def model_fit(self):
        print(requests.get("http://127.0.0.1:8000/"+self.name+"/fit").text)

    def show_rewards(self):
        plt.plot(self.rewards)
        #plt.plot([max(self.rewards[:i]) for i in range(1,len(self.rewards))])
        plt.show()

    def draw_screen(self, snake):
        data = self.drawing_data
        block_size = data['block_size']
        screen = data['screen']
        SCREEN_WIDTH, SCREEN_HEIGHT = data['SCREEN']
        color = data['color']

        screen.fill(color['BG'])

        d=snake.heading
        d_list = [np.array([0, 1]),
                  np.array([-1, 0]),
                  np.array([0, -1]),
                  np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])]

        d_l, d_s, d_r = d_list[d:d + 3]

        for i in range(SCREEN_WIDTH // block_size):
            pygame.draw.line(screen, color["line"], ((i + 0.5) * block_size, 0),
                             ((i + 0.5) * block_size, SCREEN_HEIGHT))
        for i in range(SCREEN_HEIGHT // block_size):
            pygame.draw.line(screen, color["line"], (0, (i + 0.5) * block_size), (SCREEN_WIDTH, (i + 0.5) * block_size))

        pygame.draw.circle(screen, (0, 100, 250), (snake.head[0] * block_size, snake.head[1] * block_size),
                           block_size / 2, )

        pygame.draw.rect(screen, (255, 0, 0), (
            snake.apple[0] * block_size - block_size / 4, snake.apple[1] * block_size - block_size / 4, block_size / 2,
            block_size / 2))

        for i in snake.tail:
            pygame.draw.circle(screen, (100, 100, 250), (i[0] * block_size, i[1] * block_size), block_size / 2.3, )
        pygame.display.update()

    def save_object(self):
        data = {}
        data['name'] = self.name
        data['ver'] = self.ver
        data['rewards'] = self.rewards
        requests.put("http://127.0.0.1:8000/"+self.name+"/save")
        with open(self.file_direct+self.name + "_obj.json", 'w', encoding='UTF-8') as f:
            dump(data, f, indent=' ', ensure_ascii=False)

    def learning(self, N, IsDraw, sub_N=10,print_state=False):
        if (IsDraw):
            self.drawing_data['screen'] = pygame.display.set_mode(self.drawing_data['SCREEN'])
            pygame.init()
        rewards = []
        for ep in range(N):
            requests.get("http://127.0.0.1:8000/"+self.name+"/update")
            Isplaying = True
            snake = Snake()
            obs = self.get_state(snake)
            action = self.act(obs)
            ep_rewards = 1
            ep_pos = []
            while Isplaying :
                if(print_state):
                    message = str(obs[0])
                    if (obs[1] == 0):
                        message += "l "
                    if (obs[2] == 0):
                        message += "s "
                    if (obs[3] == 0):
                        message += "r "
                    if (obs[4] == 1):
                        message += "apple l "
                    if (obs[5] == 1):
                        message += "apple u "
                    if (obs[6] == 1):
                        message += "apple r "
                    if (obs[7] == 1):
                        message += "apple d "
                    if (obs[8] == 1):
                        message += "tail l "
                    if (obs[9] == 1):
                        message += "tail u "
                    if (obs[10] == 1):
                        message += "tail r "
                    if (obs[11] == 1):
                        message += "tail d "
                    if (obs[12] == 1):
                        message += "tail front"
                    print(message)
                reward = snake.step(action - 1)
                ep_pos.append(tuple(snake.head))
                #print(ep_pos)
                ep_rewards += reward
                if(reward == 1): reward = 10
                elif (reward == -1): reward,Isplaying = -5, False
                else: reward = -0.1

                next_obs = self.get_state(snake)
                next_action = self.act(next_obs)

                self.memorize(tuple(obs), int(action), reward, tuple(next_obs))


                obs = next_obs
                action = next_action
                if (IsDraw):
                    pygame.time.Clock().tick(tick)
                    self.draw_screen(snake)

                #반복 제거
                for i in range(4,5+snake.score*2):
                    try:
                        r1=ep_pos[-i:]
                        r2=ep_pos[-1*i*2:-1*i]
                        r3=ep_pos[-1*i*3:-1*i*2]
                        if(r1==r2 and r2==r3 and len(ep_pos)>i*4 and r1[0]!=r1[1]):
                            print('Too rotate')
                            self.model_fit()
                            requests.get("http://127.0.0.1:8000/"+self.name+"/update")
                    except: 0

            self.model_fit()

            rewards.append(snake.score)
            self.rewards += rewards
            rewards.clear()
            print("step:", ep,'-',round(ep / N, 2),'%',
                  " max:", max(self.rewards),
                  " score_ep:", ep_rewards,
                  "  ep:", requests.get("http://127.0.0.1:8000/"+self.name+"/epsilon").text)

            self.save_object()

            if (ep + 1) % sub_N == 0:
                #self.make_q_table()
                self.model_fit()
        print('max_of_rewards :', max(self.rewards))
        if (IsDraw):
            pygame.quit()

    def get_state2(self, snake):
        h = np.array(snake.head)
        a = np.array(snake.apple)
        t = snake.tail
        t_l = np.array(t[0])
        d = snake.heading
        map = np.array(snake.map_size)
        d_list = [np.array([0, 1]),
                  np.array([-1, 0]),
                  np.array([0, -1]),
                  np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])]

        d_l, d_s, d_r = d_list[d:d + 3]

        state = []

        # direction l u r d tf
        state += [d]

        # not danger l s r in 2   0/1      3
        state += [
            [True, True] == list((np.zeros(2) <= h + 2 * d_l) * (h + 2 * d_l < map)) and not (
                    list(h + 2 * d_l) in t) and not (list(h + 1 * d_l) in t),
            [True, True] == list((np.zeros(2) <= h + 2 * d_s) * (h + 2 * d_s < map)) and not (
                    list(h + 2 * d_s) in t) and not (list(h + 1 * d_s) in t),
            [True, True] == list((np.zeros(2) <= h + 2 * d_r) * (h + 2 * d_r < map)) and not (
                    list(h + 2 * d_r) in t) and not (list(h + 1 * d_r) in t)]

        # not danger l s r in 1   0/1      3
        state += [
            [True, True] == list((np.zeros(2) <= h + d_l) * (h + d_l < map)) and not (list(h + 1 * d_l) in t),
            [True, True] == list((np.zeros(2) <= h + d_s) * (h + d_s < map)) and not (list(h + 1 * d_s) in t),
            [True, True] == list((np.zeros(2) <= h + d_r) * (h + d_r < map)) and not (list(h + 1 * d_r) in t)]

        # apple  l u r d
        state += list(a < h) + list(a > h)

        # tail_last l u r d
        state += list(t_l < h) + list(t_l > h)

        # Is tail front tf
        sub = np.array(h)
        tf = False
        while (([True, True] == list((np.zeros(2) <= sub) * (sub < map)))):
            if (list(sub) in t):
                tf = True
                break
            else:
                sub += d_s

        state.append(tf)

        return [int(w) if type(w) == bool else int(w) for w in state]

    def get_state1(self, snake):

        board = np.zeros((snake.map_size[0]+3)*(snake.map_size[1]+3)).reshape((snake.map_size[0]+3,snake.map_size[1]+3))
        board[0,:] = -1
        board[-1,:] = -1
        board[:,0] = -1
        board[:,-1] = -1
        board[snake.head[0]+1][snake.head[1]+1] = snake.heading+1
        for t in snake.tail:
            board[t[0]+1][t[1]+1] = 5
        if(snake.heading==0)and(snake.apple[0]<=snake.head[0]): board[snake.apple[0]][snake.apple[1]] = 6
        elif(snake.heading==1)and(snake.apple[1]<=snake.head[1]): board[snake.apple[0]][snake.apple[1]] = 6
        elif (snake.heading == 2) and (snake.apple[0] >= snake.head[0]): board[snake.apple[0]][snake.apple[1]] = 6
        elif (snake.heading == 1) and (snake.apple[1] >= snake.head[1]): board[snake.apple[0]][snake.apple[1]] = 6

        return board.tolist()

    def get_state(self, snake):
        board = np.zeros((snake.map_size[0]+3)*(snake.map_size[1]+3)).reshape((snake.map_size[0]+3,snake.map_size[1]+3))
        board[0,:] = -1
        board[-1,:] = -1
        board[:,0] = -1
        board[:,-1] = -1
        board[snake.head[0]+1][snake.head[1]+1] = snake.heading+1
        for t in snake.tail:
            board[t[0]+1][t[1]+1] = 5
        board[snake.apple[0]][snake.apple[1]] = 6

        return board.tolist()



agent = DQLearner("DQ2")
dic = {'input':[23,23],
       'output':3,
       'hidden':[ [ [8,[7,7]],[4,[3,3]],[1,[5,5]]],[256,32,32]]}

print(requests.post("http://127.0.0.1:8000/DQ2/init",headers=headers,data=dumps(dic)).text)
agent.show_rewards()
agent.learning(3000,1)
#agent.save_object()

'''
for i in range(100):
    try:
        agent = DQLearner("testCNN")
        dic = {'input':[30,30],
               'output':3,
               'hidden':[[[8,(5,5)]],[32,16]]}
        print(requests.get("http://127.0.0.1:8000/testCNN/init/",data=dumps(dic)).text)
        agent.learning(1000000,0)
        agent.save_object()
    except: print('error')
'''
