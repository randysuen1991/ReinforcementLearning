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



class ReinforcementLearningModel():
    def __init__(self,states,actions,env,episodes_size,features_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05):
        self.states = states
        self.actions = actions
        self.env = env
        self.episodes_size = episodes_size
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.actions_size = len(self.actions)
        self.epsilon = epsilon
        self.features_size = features_size
        
        
    def Predict(self):
        raise NotImplementedError
    
    def Fit(self):
        raise NotImplementedError
        
    def _Learn(self):
        raise NotImplementedError
        
    

