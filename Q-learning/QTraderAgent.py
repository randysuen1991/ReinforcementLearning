import numpy as np




class QTraderAgent():
    def __init__(self):
        super().__init__()
        
    def Act(self, state):
        """
        Acting Policy of the QTrader.
        """
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon:
            action[random.randrange(self.action_size)] = 1
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.brain.predict(state)
            action[np.argmax(act_values[0])] = 1
        return action