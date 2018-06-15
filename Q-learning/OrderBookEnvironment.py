import numpy as np



class OrderBookEnv():
    
    _actions = {
        'hold': np.array([1, 0, 0]),
        'buy': np.array([0, 1, 0]),
        'sell': np.array([0, 0, 1])
    }

    _positions = {
        'flat': np.array([1, 0, 0]),
        'long': np.array([0, 1, 0]),
        'short': np.array([0, 0, 1])
    }
    
    def __init__(self):
        pass
    
    
    def step(self,action):
        """
        Take an action (buy/sell/hold) and computes the immediate reward.
        Args:
            action (numpy.array): Action to be taken, one-hot encoded.
        Returns:
            tuple:
                - observation (numpy.array): Agent's observation of the current environment.
                - reward (float) : Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
                - info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        assert any([(action == x).all() for x in self._actions.values()])
        self._action = action
        self._iteration += 1
        done = False
        info = {}
        if all(action == self._actions['buy']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['long']
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask
            elif all(self._position == self._positions['short']):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[1]  # Ask
                instant_pnl = self._entry_price - self._exit_price
                self._position = self._positions['flat']
                self._entry_price = 0
        elif all(action == self._actions['sell']):
            reward -= self._trading_fee
            if all(self._position == self._positions['flat']):
                self._position = self._positions['short']
                self._entry_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
            elif all(self._position == self._positions['long']):
                self._exit_price = calc_spread(
                    self._prices_history[-1], self._spread_coefficients)[0]  # Bid
                instant_pnl = self._exit_price - self._entry_price
                self._position = self._positions['flat']
                self._entry_price = 0
        
        return observation, reward, done, info