import numpy as np
import pandas as pd
import random 
import copy
import sys

if 'C:\\Users\\randysuen\\Neural-Network' or 'C:\\Users\\randysuen\\Neural-Network' not in sys.path :
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')


if 'C:\\Users\\randysuen\\Reinforcement-Learning' or 'C:\\Users\\randysuen\\Reinforcement-Learning' not in sys.path :
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')

import ReinforcementLearningModel as RLM

class QTable():
    def __init__(self,states,actions):
        self.QTable = pd.DataFrame(index=states,columns=actions)        



class QLearning(RLM.ReinforcementLearningModel):
    def __init__(self,states,actions,env,episodes_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05):
        super().__init__(states,actions,env,episodes_size,decay_rate,learning_rate,epsilon)
        self.Q = QTable(states,actions)
        
    def Fit(self):
        for _ in range(self.episodes_size):
            done = False
            state = random.choice(self.states)
            while True :
                if done :
                    break
                action = self.Predict(state,self.epsilon)
                new_state, r, done, _  = self.env.Step(action)
                new_action = self.Predict(state)
                self.Q[state,action] = self.Q[state,action] + self.learning_rate * (r + self.decay_rate * self.Q[new_state,new_action] - self.Q[state,action])
                state = new_state
                
    
    def Predict(self, state, epsilon=None):
        action = max(self.Q[state],key=self.Q[state].get)
        try :
            if np.random.uniform(0,1) < 1 - epsilon + epsilon / len(self.actions) :
                return action
            else :
                actions_list = copy.copy(self.actions)
                actions_list.remove(action)
                return random.choice(actions_list)
        except :
            return action
    
class Sarsa(RLM.ReinforcementLearningModel):
    def __init__(self,states,actions,env,episodes_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05,epsilon_increment=None):
        super().__init__(states,actions,env,episodes_size,decay_rate,learning_rate,epsilon,epsilon_increment)
    