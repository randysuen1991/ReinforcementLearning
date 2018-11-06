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
        self.state = None
        self.new_state = None
        self.reward = None
        if default:
            self._construct_default_model()
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())

    def fit(self):
        for i in range(self.env.episodes_size):
            step = 0
            state = self.env.reset()
            while True:
                action = self.predict(state)
                self.env.actions.append(action)
                # In cases like financial environments, the action would give no impact to the result of the next step.
                new_state, reward, done = self.env.step()
                self._learn(state, reward, new_state, action)
                state = new_state
                step += 1
                if done:
                    break

    def predict(self, state):
        probs = self.sess.run(fetches=self.actor_model.output, feed_dict={self.actor_model.input:state})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def _construct_default_model(self):
        with self.graph.as_default():
            with tf.variable_scope('actor'):
                self.actor_model = NNM.NeuralNetworkModel(graph=self.graph)
                self.actor_model.td_error = tf.placeholder(dtype=self.dtype, shape=None)
                self.actor_model.action = tf.placeholder(dtype=self.dtype, shape=None)
                self.actor_model.build(NNU.NeuronLayer(hidden_dim=20), input_dim=self.env.features_dim)
                self.actor_model.build(NNU.NeuronLayer(hidden_dim=self.env.actions_num))
                self.actor_model.compile(optimizer=tf.train.GradientDescentOptimizer,
                                         loss_fun=NNL.NeuralNetworkLoss.exploss,
                                         action=self.actor_model.action,
                                         td_error=self.actor_model.td_error)

            with tf.variable_scope('critic'):
                self.critic_model = NNM.NeuralNetworkModel(graph=self.graph)
                self.critic_model.reward = tf.placeholder(dtype=self.dtype, shape=None)
                self.critic_model.value = tf.placeholder(dtype=self.dtype, shape=[1, 1])
                self.critic_model.build(NNU.NeuronLayer(hidden_dim=10), input_dim=self.env.features_dim)
                self.critic_model.build(NNU.NeuronLayer(hidden_dim=1))
                self.critic_model.compile(optimizer=tf.train.GradientDescentOptimizer,
                                          loss_fun=NNL.NeuralNetworkLoss.tdsquared,
                                          reward=self.critic_model.reward,
                                          gamma=self.gamma)

    def _learn(self, state, reward, new_state, action):
        # critic model learns first.
        value = self.sess.run(fetches=self.critic_model.output, feed_dict={self.critic_model.input: new_state})
        td_error, _ = self.sess.run(fetches=[self.critic_model.loss, self.critic_model.train],
                                    feed_dict={self.critic_model.input: state,
                                               self.critic_model.value: value,
                                               self.reward: reward})
        loss, _ = self.sess.run(fetches=[self.actor_model.loss, self.actor_model.train],
                                feed_dict={self.actor_model.input: state,
                                           self.actor_model.action: action,
                                           self.actor_model.td_error: td_error})
        return loss
