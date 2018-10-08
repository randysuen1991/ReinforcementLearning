"""
This file consists three parts:
    The first is the action-value function, Q, and state-value function, V, for policy pi.
    The second one is the learning algorithm, including SARSA, Q-learning, .... .
    The last is the model itself which uses the learning algorith to make decision.
    
"""

from abc import ABC, abstractmethod
# This class should be a double keys one value dictionary.


class ReinforcementLearningModel(ABC):
    def __init__(self, env, gamma=0.8, decay_rate=0.1, learning_rate=0.01, epsilon=0.05):
        self.env = env
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.actions_size = len(self.env.actions_space)
        self.epsilon = epsilon
        # It would be specified when 'Fit' attribute is called.
        self.gamma = gamma
        
    @abstractmethod
    def predict(self, state, epsilon=None):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def _learn(self):
        raise NotImplementedError
        
    

