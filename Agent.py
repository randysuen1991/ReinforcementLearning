import numpy as np


class MetaAgent(type):
    def __new__(meta,name,bases,class_dict):
        return type.__new__(meta,name,bases,class_dict)

class AbstractAgent(meta=MetaAgent):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.states_size = len(states)
        self.actions_size = len(actions)
        self.current_state = None
        self._iter = 0
    
    def Reset(self):
        pass
    
    def Step(self,state):
        raise NotImplementedError
        
# This is a abstract class for the agent.
class Agent(AbstractAgent):
    def __init__(self,states,actions):
        super().__init__(states,actions)
        
    # This function should return the intial state of an episode.
    def Reset(self):
        pass
    
    # 
    def Step(self,action):
        pass
    
    
        
    
        
    