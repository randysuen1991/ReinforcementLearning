import numpy as np

# This is a abstract class for the agent.
class Agent():
    
    def __init__(self, state_size, action_size, model, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.epsilon = epsilon
    
    def Act(self,state):
        raise NotImplementedError
        
    
        
    