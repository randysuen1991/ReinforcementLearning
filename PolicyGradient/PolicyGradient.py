import tensorflow as tf
import ReinforcementLearning.ReinforcementLearningModel as RLM


class PolicyGradient(RLM.ReinforcementLearningModel):
    def __init__(self, env, gamma=0.8, decay_rate=0.1, learning_rate=0.01, epsilon=0.05):
        super().__init__(env, gamma=gamma, decay_rate=decay_rate, learning_rate=learning_rate, epsilon=epsilon)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

    def fit(self):
        pass

    def _learn(self):
        pass

    def _construct_default_models(self):
        pass

    def predict(self, state, epsilon=None):
        pass
