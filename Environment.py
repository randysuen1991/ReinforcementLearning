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
    def __init__(self,filename,history_prices,states_type=0,window_size=10):
        super().__init__()
        self.prices = list()
        self.dates = list()
        self.workbook = xlrd.open_workbook(filename)
        self.sheets_name = self.workbook.sheet_names
        self.window_size = window_size
        self.sheets_count = 0
        self.states_type = states_type
        
            
    def Reset(self) :
        
        try :    
            self.sheet = self.workbook[self.sheets_name[self.sheets_count]]
        except IndexError as E :
            raise E('No more episodes!')
            
        self.sheets_count += 1
        self.days_size = self.sheet.nrows - 1
        self.iteration = self.days_size
        self._Default()
        
    def Step(self,action):
        if self.iteration == 1 :
            self.iteration -= 1
            return self._Read_Daily(), self._Reward(action), True
        else :
            self.iteration -= 1
            return self._Read_Daily(), self._Reward(action), False
    
    def _Default(self) :
        for i in range(self.window_size):    
            self.prices.append(self.sheet.row_values(self.days_size-i)[1])
            self.dates.append(self.sheet.row_values(self.days_size-i)[0])
            self.iteration -= 1
    
    def _Reward(self, action):
        window_price = self.prices[-(self.window_size+1)]
        sign = np.sign(action)
        current_price = self.prices[-1]
        last_price = self.prices[-2]
        return (1+sign*(current_price-last_price)/last_price)*(last_price/window_price)
    
    def _State(self):
        if self.states_type == 0 :
            pass
        elif self.states_type == 1 :
            pass
        elif self.states_type == 2 :
            pass
        else :
            raise NotImplementedError('Please specify correct states type.')
    # Return the daily closing price.
    def _Read_Daily(self) :
        self.iteration -= 1
        row_data = self.sheet.row_values(self.iteration)
        self.dates.append(row_data[0])
        self.prices.append(row_data[1])
        
        
if __name__ == '__main__' :
    filename = 'SP500.xlsx'
    bk = xlrd.open_workbook(filename)
    print(bk.sheet_names())
#    sh = bk.sheet_by_name('sheet1')
#    for i in range(sh.nrows):
#        print(i)
#        print(sh.row_values(i))
        