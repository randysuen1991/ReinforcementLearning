import sys


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
    def __init__(self, env, episodes_size, features_size, gamma=0.8, decay_rate=0.1, learning_rate=0.01, epsilon=0.05):
        self.env = env
        self.episodes_size = episodes_size
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.actions_size = len(self.env.actions_space)
        self.epsilon = epsilon
        # It would be specified when 'Fit' attribute is called.
        self.features_size = features_size
        self.gamma = gamma
        
    def Predict(self):
        raise NotImplementedError
    
    def Fit(self):
        raise NotImplementedError
        
    def _Learn(self):
        raise NotImplementedError
        
    

