import tensorflow as tf
import numpy as np
import ReinforcementLearning.Core.ReinforcementLearningModel as RLM


class ActorCritic(RLM.ReinforcementLearningModel):
    def __init__(self, env, gamma=0.8, batch_size=40, decay_rate=0.1, learning_rate=0.01, epsilon=0.05, default=True):
        super().__init__(env, gamma=gamma, decay_rate=decay_rate, learning_rate=learning_rate, epsilon=epsilon)
        self.batch_size = batch_size
        self.graph = tf.Graph()
        self.sess = tf.Session(self.graph)
        if default:
            self._construct_default_model()
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())

    def fit(self):
        pass

    def predict(self, state, epsilon=None):
        pass

    def _construct_default_model(self):
        pass
    
    def _learn(self):
        pass
