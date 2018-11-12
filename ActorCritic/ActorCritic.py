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
        probs = self.sess.run(fetches=self.actor_model.output, feed_dict={self.actor_model.input: state})
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

    def _learn(self, state, action, reward, new_state):
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


class DeepDeterministicPolicyGradient(ActorCritic):
    def __init__(self, env, gamma=0.8, batch_size=40, decay_rate=0.1, learning_rate=0.01, epsilon=0.05,
                 dtype=tf.float32, default=True, capacity=30):
        super().__init__(env, gamma=gamma, decay_rate=decay_rate, learning_rate=learning_rate, epsilon=epsilon,
                         dtype=dtype, default=default, batch_size=batch_size)
        self.memory_counter = 0
        self.capacity = capacity
        self.memory = np.zeros((capacity, 2 * self.env.features_dim + self.env.actions_num + 1))
        self.actor_e_params = None
        self.actor_t_params = None
        self.critic_e_params = None
        self.critic_t_params = None

    def fit(self):
        for i in range(self.env.episodes_size):
            step = 0
            state = self.env.reset()
            while True:
                action = self.predict(state)
                self.env.actions.append(action)
                # In cases like financial environments, the action would give no impact to the result of the next step.
                new_state, reward, done = self.env.step()
                self._store_transition(state, action, reward, new_state)
                if self.memory_counter > self.capacity:
                    samples = self._sample(self.batch_size)
                    s_state = samples[:, :self.env.features_dim]
                    s_action = samples[:, self.env.features_dim: self.env.features_dim + self.env.actions_num]
                    s_reward = samples[:, -self.env.features_dim - 1: -self.env.features_dim]
                    s_new_state = samples[:, -self.env.features_dim:]
                    self._learn(s_state, s_action, s_reward, s_new_state)
                state = new_state

                step += 1
                if done:
                    break

    def _construct_default_model(self):
        with self.graph.as_default():
            with tf.variable_scope('actor'):
                with tf.variable_scope('eval'):
                    self.actor_eval_model = NNM.NeuralNetworkModel(graph=self.graph)
                    self.actor_eval_model.build(NNU.NeuronLayer(hidden_dim=30, trainable=True),
                                                input_dim=self.env.features_dim)
                    self.actor_eval_model.build(NNU.NeuronLayer(hidden_dim=self.env.actions_num, trainable=True))

                with tf.variable_scope('targ'):
                    self.actor_targ_model = NNM.NeuralNetworkModel(graph=self.graph)
                    self.actor_targ_model.build(NNU.NeuronLayer(hidden_dim=30, trainable=False),
                                                input_dim=self.env.features_dim)
                    self.actor_targ_model.build(NNU.NeuronLayer(hidden_dim=self.env.actions_num, trainable=False))

                self.actor_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/eval')
                self.actor_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/targ')

            with tf.variable_scope('critic'):
                with tf.variable_scope('eval'):
                    self.critic_eval_model_state = NNM.NeuralNetworkModel(graph=self.graph)
                    self.critic_eval_model_state.build(NNU.NeuronLayer(hidden_dim=10, trainable=True),
                                                       input_dim=self.env.features_dim)
                    self.critic_eval_model_action = NNM.NeuralNetworkModel(graph=self.graph)
                    self.critic_eval_model_action.build(NNU.NeuronLayer(hidden_dim=10, trainable=True),
                                                        input_dim=self.env.features_dim)

                    self.critic_eval_model_action.input = self.actor_eval_model.output

                    # integrate all the models into one.
                    self.critic_eval_model = self.critic_eval_model_action + self.critic_eval_model_state
                    self.critic_eval_model.input_state = self.critic_eval_model_state.input
                    self.critic_eval_model.input_action = self.critic_eval_model_action.input

                with tf.variable_scope('targ'):
                    self.critic_targ_model_state = NNM.NeuralNetworkModel(graph=self.graph)
                    self.critic_targ_model_state.build(NNU.NeuronLayer(hidden_dim=10, trainable=False),
                                                       input_dim=self.env.features_dim)
                    self.critic_targ_model_action = NNM.NeuralNetworkModel(graph=self.graph)
                    self.critic_targ_model_action.build(NNU.NeuronLayer(hidden_dim=10, trainable=False),
                                                        input_dim=self.env.features_dim)

                    self.critic_targ_model_state.action = self.actor_targ_model.output

                    # integrate all the models into one.
                    self.critic_targ_model = self.critic_targ_model_action + self.critic_targ_model_state
                    self.critic_targ_model.input_state = self.critic_targ_model_state.input
                    self.critic_targ_model.input_action = self.critic_targ_model_action.input

                self.critic_e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/eval')
                self.critic_t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/targ')

                with tf.variable_scope('a_grad'):
                    pass
                with tf.variable_scope(''):
                    pass

            # Connect the actor to the critic.
            with tf.variable_scope('policy_grads'):
                # dq/da * da/dp
                self.policy_grads = tf.gradient(ys=self.actor_eval_model.output, xs=self.e_params, )

    def predict(self, state):
        probs = self.sess.run(fetches=self.actor_model.output, feed_dict={self.actor_model.input: state})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

    def _store_transition(self, state, action, reward, new_state):
        transition = np.hstack((state, action, reward, new_state))
        index = self.memory_counter % self.capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def _sample(self, n):
        indices = np.random.choice(self.capacity, size=n)
        return self.memory[indices, :]
