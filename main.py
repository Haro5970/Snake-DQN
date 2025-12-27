from flask import Flask
from flask import request
import DQN_class
from keras.models import Sequential
from keras.layers import Dense,Input

api = Flask(import_name=__name__)

model = {}

@api.route('/<name>/init', methods=['POST'])
def Init(name): #name,input,output,hidden
    global model
    dic = request.get_json()
    #print(dic)
    model[name] = DQN_class.DQN_Agent(name,dic['input'],dic['output'],dic['hidden'])

    #print('init')
    return str(model[name])


@api.route('/<name>/memorize', methods=['PUT'])
def Memorize(name):
    global model
    dic = request.get_json()
    model[name].memorize(dic['state'],dic['action'],dic['reward'],dic['next_state'])
    #print('memorize')
    return "memorize"


@api.route('/<name>/act', methods=['POST'])
def Act(name):
    global model
    #print('act')
    return str(model[name].act(request.get_json()['state']))

@api.route('/<name>/fit', methods=['GET'])
def Fit(name):
    global model
    r = model[name].model_fitting()
    #print("fit")
    return r

@api.route('/<name>/save', methods=['PUT'])
def Save(name):
    global model
    model[name].save_data()
    #print('save')
    return "save"


@api.route('/<name>/update', methods=['GET'])
def Model_update(name):
    global model
    model[name].update_model()
    #print('model update')
    return "model update"


@api.route('/<name>/epsilon', methods=['GET'])
def Get_epsilon(name):
    global model
    #print(epsilon)
    return str(model[name].epsilon)


if __name__ == '__main__':
    api.run(debug=True, port=8000)
