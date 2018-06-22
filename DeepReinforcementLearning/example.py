import DeepReinforcementLearningModel as DRLM
import MarketTradingEnv as MTE


def example1():
    mtenv = MTE.MarketTradingEnv(filename='SP500.xlsx')
    agent = DRLM.DeepQLearning()
    


if __name__ == '__main__' :
    example1()