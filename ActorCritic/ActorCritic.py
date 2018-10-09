import tensorflow as tf
import numpy as np
from NeuralNetwork import NeuralNetworkModel as NNM
from NeuralNetwork import NeuralNetworkUnit as NNU
from NeuralNetwork import NeuralNetworkLoss as NNL
import ReinforcementLearning.Core.ReinforcementLearningModel as RLM


class ActorCritic(RLM.ReinforcementLearningModel):
    def __init__(self, env, gamma=0.8, batch_size=40, decay_rate=0.1, learning_rate=0.01, epsilon=0.05,
                 dtype=tf.float32, default=True):
        super().__init__(env, gamma=gamma, decay_rate=decay_rate, learning_rate=learning_rate, epsilon=epsilon)
        self.batch_size = batch_size
        self.graph = tf.Graph()
        self.sess = tf.Session(self.graph)
        self.dtype = dtype
        if default:
            self._construct_default_model()
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())

    def fit(self):
        pass

    def predict(self, state, epsilon=None):
        pass

    def _construct_default_model(self):
        with self.graph.as_default():
            with tf.variable_scope('actor'):
                self.actor_model.td_error = tf.placeholder(dtype=self.dtype, shape=None)
                self.actor_model = NNM.NeuralNetworkModel(graph=self.graph)
                self.actor_model.build(NNU.NeuronLayer(hidden_dim=20), input_dim=self.env.features_dim)
                self.actor_model.build(NNU.NeuronLayer(hidden_dim=self.env.actions_num))

            with tf.variable_scope('critic'):
                self.critic_model.reward = tf.placeholder(dtype=self.dtype, shape=None)
                self.critic_model = NNM.NeuralNetworkModel(graph=self.graph)
                self.critic_model.build(NNU.NeuronLayer(hidden_dim=10), input_dim=self.env.features_dim)
                self.critic_model.build(NNU.NeuronLayer(hidden_dim=1))
                self.critic_model.compile(optimizer=tf.train.GradientDescentOptimizer,
                                          loss_fun=NNL.NeuralNetworkLoss.tdsquared,
                                          reward=self.critic_model.reward,
                                          gamma=self.gamma)
                



    def _learn(self):
        pass
