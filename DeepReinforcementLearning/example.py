import DeepReinforcementLearningModel as DRLM
import MarketTradingEnv as MTE

# Market Trading using S%P500 example.
def example1():
    
    mtenv = MTE.MarketTradingEnv(filename='SP500.xlsx',features_size=8,rewards_type=1)
    agent = DRLM.DeepQLearning(env=mtenv,memory_size=40,batch_size=10,gamma=0.2,epsilon=0.6)
    agent.Fit()
    mtenv.Plot(action_type=1,low = 2000,high=2500)
    mtenv.OneLotAccumulatedProfit(target=1,low=None,high=None)
    
def example2():
    

    pass



if __name__ == '__main__' :
    example1()