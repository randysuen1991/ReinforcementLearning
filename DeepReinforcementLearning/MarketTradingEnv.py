import numpy as np
import xlrd

import sys
if 'C:\\Users\\randysuen\\Reinforcement-Learning' or 'C:\\Users\\randysuen\\Reinforcement-Learning' not in sys.path :
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')

import Environment as E


# This is an environment for trading the S%P500.
class MarketTradingEnv(E.Environment):
    
    actions_space = [1,0,-1]
    
    def __init__(self,filename,features_size=None,rewards_type=0,window_size=10):
        super().__init__()
        self.prices = list()
        self.dates = list()
        self.workbook = xlrd.open_workbook(filename)
        # This attribute is a list of names of episodes.
        self.episodes = self.workbook.sheet_names()
        self.window_size = window_size
        self.features_size = self.window_size if features_size == None else features_size 
        assert self.features_size <= self.window_size 
        self.episodes_count = 0
        self.rewards_type = rewards_type
    
    
    # This method is used to synchronize the output of the model and the expected action of the environment.
    def DealAction(self,action):
        # specify the results
        if action == 0 :
            action = -1
        elif action == 1 :
            action = 0
        elif action == 2 :
            action = 1
        else :
            raise ValueError('Too many actions in the evaluation model.')
        
        return action
        
    def Reset(self) :
        try :    
            self.episode = self.workbook.sheet_by_name(self.episodes[self.episodes_count])
        except IndexError as E :
            raise E('No more episodes!')
            
        
        self.episodes_count += 1
        self.days_size = self.episode.nrows - 1
        self.iteration = self.days_size
        
        for i in range(self.window_size):    
            row_values = self.episode.row_values(self.days_size-i)
            self.prices.append(row_values[1])
            self.dates.append(row_values[0])
            self.iteration -= 1
        
        self._Read_Daily()
        return np.array([self._State()])
    
    def Step(self,action):
        
        assert action in self.actions_space
        
        self._Read_Daily()
        if self.iteration == 1 :
            return np.array([self._State()]), self._Reward(action), True
        else :
            return np.array([self._State()]), self._Reward(action), False
        
        
    # In this case, the states are independent of the action.
    def _State(self):
        state = list()
        for i in range(self.features_size):
            state.append(self.prices[-1-i]-self.prices[-1-i-1])
        return state
    
    def _Reward(self, action):
        if self.rewards_type == 0 :
            return self.prices[-1] - self.prices[-2]
        elif self.rewards_type == 1 :
            return self.prices[-1]/self.prices[-2] - 1
        elif self.rewards_type == 2 :
            return (1+np.sign(action)*((self.prices[-1]-self.prices[-2])/self.prices[-2]))*(self.prices[-2]/self.prices[-1-self.window_size]) 
        else :
            raise NotImplementedError('Please specify correct states type.')
    # Return the daily closing price.
    def _Read_Daily(self) :
        row_data = self.episode.row_values(self.iteration)
        self.dates.append(row_data[0])
        self.prices.append(row_data[1])
        self.iteration -= 1
        