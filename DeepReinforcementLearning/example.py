import DeepReinforcementLearningModel as DRLM
import MarketTradingEnv as MTE


def example1():
    mtenv = MTE.MarketTradingEnv(filename='SP500.xlsx')
    agent = DRLM.DeepQLearning(env=mtenv,memory_size=50,features_size=10,batch_size=10)
    agent.Fit()


if __name__ == '__main__' :
    example1()