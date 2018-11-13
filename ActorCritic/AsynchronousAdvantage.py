import tensorflow as tf
from ReinforcementLearning.ActorCritic import ActorCritic
import multiprocessing as mp


class ACNet:
    def __init__(self, globalnet=None):
        if globalnet is None:
            pass
        else:
            pass

    def _build_net(self):
        pass


class Worker:

    def __init__(self, name, globalAC):
        pass

    def work(self):
        pass


class A3C(ActorCritic.ActorCritic):

    def __init__(self, env, gamma=0.8, batch_size=40, decay_rate=0.1, learning_rate=0.001, epsilon=0.05,
                 dtype=tf.float32, default=True, capacity=30):
        super().__init__(env, gamma=gamma, decay_rate=decay_rate, learning_rate=learning_rate, epsilon=epsilon,
                         dtype=dtype, default=default, batch_size=batch_size)
