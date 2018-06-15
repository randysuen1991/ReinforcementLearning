import numpy as np
import random 
import copy
import sys
import pandas as pd

if 'C:\\Users\\randysuen\\Neural-Network' or 'C:\\Users\\randysuen\\Neural-Network' not in sys.path :
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')
    
    
"""
This file consists three parts:
    The first is the action-value function, Q, and state-value function, V, for policy pi.
    The second one is the learning algorithm, include SARSA, Q-learning, .... .
    The last is the model itself which uses the learning algorith to make decision.
    
"""

# This class should be a double keys one value dictionary.
class QTable():
    def __init__(self,states,actions):
        self.QTable = pd.DataFrame(index=states,columns=actions)        

# The following two functions return Multikeys object.
# The input argument states and actions should be two lists. 
def Q_Generator(states,actions,random=True):
    if random :
        return MultikeysDict(states,actions,init=True)
    
    
def V_Generator(state_size,random=True):
    if random :
        return np.random.rand(state_size,1)
    


class ReinforcementLearningModel():
    def __init__(self,states,actions,env,episodes_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05):
        self.states = states
        self.actions = actions
        self.env = env
        self.episodes_size = episodes_size
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.Q = Q_Generator(states,actions)
        
    def Predict(self):
        raise NotImplementedError
    
    def Fit(self):
        raise NotImplementedError
    
    
class QLearning(ReinforcementLearningModel):
    def __init__(self,states,actions,env,episodes_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05):
        super().__init__(states,actions,env,episodes_size,decay_rate,learning_rate,epsilon)
        
        
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
            if np.random.uniform(0,1) < 1 - epsilon + epsilon/len(self.actions) :
                return action
            else :
                actions_list = copy.copy(self.actions)
                actions_list.remove(action)
                return random.choice(actions_list)
        except :
            return action
    
class Sarsa(ReinforcementLearningModel):
    def __init__(self,states,actions,env,episodes_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05):
        super().__init__(states,actions,env,episodes_size,decay_rate,learning_rate,epsilon)
    
class DeepQLearning(ReinforcementLearningModel):
    def __init__(self,states,actions,env,episodes_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05):
        super().__init__(states,actions,env,episodes_size,decay_rate,learning_rate,epsilon)
    
    