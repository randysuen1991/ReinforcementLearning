import numpy as np


class MetaAgent(type):
    def __new__(meta,name,bases,class_dict):
        return type.__new__(meta,name,bases,class_dict)

class AbstractAgent(meta=MetaAgent):
    pass
# This is a abstract class for the agent.
class Agent(AbstractAgent):
    
    def __init__(self, state_size, action_size, model, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model
        self.epsilon = epsilon
    
    def Act(self,state):
        raise NotImplementedError
        
    
        
    