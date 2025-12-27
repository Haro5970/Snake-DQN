from collections import deque
import numpy as np
from json import dump, load
import random as rd
from keras.layers import Dense,Input, Conv2D,MaxPooling2D,Flatten
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from tqdm import tqdm
from pprint import pprint
import os



class DQN_Agent:
    def __init__(self,model_name,state_size=10,act_size=3,hidden_layer=[],memory_size=4096,learning_rate=0.01,gamma=0.95,ep_start_end=(1.0,0.001)):
        self.model_name = model_name
        self.file_dir = "models/"+model_name+'/'
        try:
            os.mkdir("models/"+model_name)
        except:
            0

        try:
            self.load_data()
            print("Load!")
        except:
            self.state_size = state_size
            self.act_size = act_size
            self.hidden_layer = hidden_layer
            self.memory_size = memory_size
            self.learning_rate = learning_rate
            self.gamma = gamma
            self.ep_start = ep_start_end[0]
            self.ep_end = ep_start_end[1]
            self.epsilon = self.ep_start
            self.memory = deque([],maxlen=self.memory_size)
        self.model_learning = self.create_model()
        self.model_target = self.create_model()
        try:
            self.model_learning.load_weights(self.file_dir+"weights.h5")
        except: 0

    def __str__(self):
        s = 'Model : ' + self.model_name+'\n'

        model_string = []
        self.model_learning.summary(print_fn=lambda x: model_string.append(x))

        while (True):
            try:
                model_string.remove("                                                                 ")
            except:
                break

        model_string = [w for w in model_string if ("Model" not in w)
                        and (" Layer (type)                Output Shape              Param #   " not in w)
                        and ("=================================================================" not in w)
                        and ("_________________________________________________________________" not in w)
                        and (w != [])][:-2]
        model_string[:-1] = ["".join(w) + "\n" for w in model_string[:-1]]
        s += "".join(model_string)+'\n'

        s += "Act : " + str(self.act_size) + '\n'
        s += "Memory Size : " + str(self.memory_size) + '\n'
        s += "Learning rate : " + str(self.learning_rate) + '\n'
        s += "Gamma : " + str(self.gamma) + '\n'
        s += "Epsilon : " + str(self.ep_start) + " -> " + str(self.epsilon) + " -> " + str(self.ep_end)
        return s

    def create_model(self):
        model = Sequential()
        if (type(self.state_size)==int):
            model.add(Input(shape=(None, self.state_size)))
            for i in self.hidden_layer:
                model.add(Dense(i, 'relu'))
            model.add(Dense(self.act_size, 'linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        elif(type(self.state_size)==list):
            model.add(Input(shape=(self.state_size[0],self.state_size[1],1)))
            for i in self.hidden_layer[0]:
                model.add(Conv2D(filters=i[0],
                                 kernel_size=i[1],
                                 strides=(1,1),
                                 padding='same',
                                 activation='relu' ))
                model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
            model.add(Flatten())
            for i in self.hidden_layer[1]:
                #print(i)
                model.add(Dense(i,'relu'))
            model.add(Dense(self.act_size,'linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def save_data(self):
        dic = {}

        dic['state_size'] = self.state_size
        dic['act_size'] = self.act_size
        dic['hidden_layer'] = self.hidden_layer
        dic['memory_size'] = self.memory_size
        dic['learning_rate'] = self.learning_rate
        dic['gamma'] = self.gamma
        dic['ep_start'] = self.ep_start
        dic['ep_end'] = self.ep_end
        dic['epsilon'] = self.epsilon
        with open(self.file_dir+"data.json",'w',encoding='utf-8')as f:
            dump(dic,f,ensure_ascii=False,indent='\t')

        with open(self.file_dir+"memory.json",'w',encoding='utf-8')as f:
            dump(list(self.memory),f,ensure_ascii=False,indent=' ')

        self.model_learning.save_weights(self.file_dir+"weights.h5")

    def load_data(self):
        with open(self.file_dir+"data.json",'r',encoding='utf-8')as f:
            dic = load(f)
        self.state_size = dic['state_size']
        self.act_size = dic['act_size']
        self.hidden_layer = dic['hidden_layer']
        self.memory_size = dic['memory_size']
        self.learning_rate = dic['learning_rate']
        self.gamma = dic['gamma']
        self.ep_start = dic['ep_start']
        self.ep_end = dic['ep_end']
        self.epsilon = dic['epsilon']

        with open(self.file_dir+"memory.json",'r',encoding='utf-8')as f:
            self.memory = deque(load(f),self.memory_size)



    def model_fitting(self,min=2,sample=16):
        # memory : [(state1,act1, reward1, next_state1) ,(state2,act2, reward2, next_state2) ...
        if (len(self.memory)<self.memory_size//min):
            return "Not Fit-"+str(len(self.memory))

        state, action, reward, next_state = [],[],[],[]
        for m in rd.sample(self.memory,self.memory_size//sample):
            state.append(m[0])
            action.append(m[1])
            reward.append(m[2])
            next_state.append(m[3])

        x = np.array(state)

        #print(x.shape)
        #print(x)
        q_value = self.model_learning.predict(x,verbose=3)
        next_q_value = np.array([ np.max(w) for w in self.model_learning.predict(np.array(next_state),verbose=3)])
        #print(np.array(reward).dtype,type(self.gamma),next_q_value.dtype)
        target = np.array(reward) + self.gamma *next_q_value
        #print(action)
        for i in range(self.memory_size//sample):
            q_value[i][action[i]] = target[i]
        hist = self.model_learning.fit(x,q_value,verbose=3)
        self.epsilon = round(self.epsilon*0.99,6) if self.epsilon>self.ep_end else self.ep_end
        #print("--Model Update--")
        return "Model Fit-"+str(round(hist.history['loss'][0],4))

    def act(self,state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.act_size)
        else:
            q_values = self.model_learning.predict(np.array([state]),verbose=3)
            action = np.argmax(q_values)
            #print(q_values)
        return action

    def memorize(self,state,action,reward,next_state):
        self.memory.append( (state,action,reward,next_state) )
        '''
        if(reward>0): self.memory.append( (state,action,reward,next_state) )
        if (reward > 0): self.memory.append((state, action, reward, next_state))
        if (reward > 0): self.memory.append((state, action, reward, next_state))
        '''

    def update_model(self):
        self.model_target.set_weights(self.model_target.get_weights())
