import numpy as np

# This is a abstract class for the agent.
class Agent():
    
    def __init__(self,
                 state_size,
                 action_size,
                 episodes
                 ):
        self.state_size = state_size
        self.action_size = action_size
    
    
    def Act(self,state):
        raise NotImplementedError
        
    
        
    