import numpy as np
import xlrd



# This is an abstract class for environment. The invironments in all the cases would follow this class. 
class Environment():
    def __init__(self):
        pass
    def Reset(self):
        pass
    def Step(self,action):
        raise NotImplementedError()
    
    
# This is an environment for trading the S%P500.
class MarketTradingEnv(Environment):
    def __init__(self,filename,states_size=None,rewards_type=0,window_size=10):
        super().__init__()
        self.prices = list()
        self.dates = list()
        self.workbook = xlrd.open_workbook(filename)
        self.sheets_name = self.workbook.sheet_names()
        self.episodes_size = len(self.sheets_name)
        self.window_size = window_size
        self.states_size = self.window_size if states_size == None else states_size 
        self.sheets_count = 0
        self.rewards_type = rewards_type
        
            
    def Reset(self) :
        try :    
            self.sheet = self.workbook.sheet_by_name(self.sheets_name[self.sheets_count])
        except IndexError as E :
            raise E('No more episodes!')
            
        
        self.sheets_count += 1
        self.days_size = self.sheet.nrows - 1
        self.iteration = self.days_size
        for i in range(self.window_size):    
            row_values = self.sheet.row_values(self.days_size-i)
            self.prices.append(row_values[1])
            self.dates.append(row_values[0])
            self.iteration -= 1

    def Step(self,action):
        
        self._Read_Daily()
        if self.iteration == 1 :
            return np.array([self._State()]), self._Reward(action), True
        else :
            return np.array([self._State()]), self._Reward(action), False
        
        
    # In this case, the states are independent of the action.
    def _State(self):
        states = list()
        for i in range(self.states_size):
            states.append(self.prices[-1-i]-self.prices[-1-i-1])
        return states
    
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
        row_data = self.sheet.row_values(self.iteration)
        self.dates.append(row_data[0])
        self.prices.append(row_data[1])
        self.iteration -= 1
        
         