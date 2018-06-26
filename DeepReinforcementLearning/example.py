import DeepReinforcementLearningModel as DRLM
import MarketTradingEnv as MTE
import sys 

if 'C:\\Users\\randysuen\\Financial-Analysis' not in sys.path :
    sys.path.append('C:\\Users\\randysuen\\Financial-Analysis')


import FinAnalysis as FA

# Market Trading using S%P500 example.
def example1():
    window_size=100
    mtenv = MTE.MarketTradingEnv(filename='SP500.xlsx',features_size=100,rewards_type=2,window_size=window_size)
    agent = DRLM.DeepQLearning(env=mtenv,memory_size=80,batch_size=64,gamma=0.85,epsilon=0.6,learning_rate=0.0001)
    agent.Fit()
#    mtenv.Plot(action_type = 1,low = 2000,high = 2500)
    FA.FinAnalysis.OneLotAccumulatedProfit(prices=mtenv.prices[window_size:],actions=mtenv.actions)
    
def example2():
    

    pass



if __name__ == '__main__' :
    example1()