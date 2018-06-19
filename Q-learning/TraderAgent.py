import numpy as np
import Agent as A






""" 
Normal version
"""
#from .. import Agent




class TraderAgent(A.Agent):
    # The model should be an ReinforcementModel type object from Model.py file.
    def __init__(self, state_size, action_size, model, epsilon):
        super().__init__(state_size, action_size, model, epsilon)
        
    def Act(self, state):
        state = state.reshape(1, self.state_size)
        act_values = self.model.Predict(state)
        action[np.argmax(act_values[0])] = 1
        return action