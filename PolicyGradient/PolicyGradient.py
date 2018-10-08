import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ReinforcementLearning.Core.ReinforcementLearningModel as RLM
import NeuralNetwork.NeuralNetworkModel as NNM
import NeuralNetwork.NeuralNetworkUnit as NNU
import NeuralNetwork.NeuralNetworkLoss as NNL


class PolicyGradient(RLM.ReinforcementLearningModel):
    def __init__(self, env, gamma=0.8, batch_size=1, decay_rate=0.1, learning_rate=0.01, epsilon=0.05, default=True):
        super().__init__(env, gamma=gamma, decay_rate=decay_rate, learning_rate=learning_rate, epsilon=epsilon)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.batch_size = batch_size
        self.episode_states = list()
        self.episode_rewards = list()
        self.episode_actions = list()
        if default:
            self._construct_default_models()
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())

    def fit(self, show_graph=True):
        for i in range(self.env.episodes_size):
            state = self.env.Reset(iteration=i)
            while True:
                action = self.predict(state, self.epsilon)
                self.env.actions.append(action)
                new_state, reward, done = self.env.step()
                if done:
                    rewards_sum = sum(self.episode_rewards)
                    if 'running_reward' not in globals():
                        running_reward = rewards_sum
                    else:
                        running_reward = running_reward * 0.99 + rewards_sum * 0.01
                    print("episode:", i, "  reward:", int(running_reward))
                value = self._learn()

                if i == 0:
                    plt.plot(value)
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break

    def _learn(self):
        discounted_episode_rewards_norm = self._discount_and_norm_rewards()
        loss, _ = self.sess.run(fetches=[self.policy_model.loss, self.policy_model.train],
                                feed_dict={self.policy_model.input: np.vstack(self.episode_states),
                                           self.policy_model.target: np.hstack(self.episode_actions),
                                           self.policy_model.action_state_value: discounted_episode_rewards_norm
                                           })
        self.episode_rewards, self.episode_actions, self.episode_states = list(), list(), list()
        return discounted_episode_rewards_norm

    def _store_transition(self, state, action, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def _construct_default_models(self):
        with self.graph.as_default():
            with tf.variable_scope('Policy_Network'):
                self.policy_model = NNM.NeuralNetworkModel(graph=self.graph)
                self.policy_model.build(NNU.NeuronLayer(hidden_dim=30, transfer_fun=tf.nn.sigmoid),
                                        input_dim=self.env.features_size)
                self.policy_model.build(NNU.NeuronLayer(hidden_dim=30, transfer_fun=None))
                self.policy_model.build(NNU.SoftMaxLayer())
                self.policy_model.action_state_value = tf.placeholder(shape=[None, ])

            self.policy_model.mini_batch = self.batch_size
            self.policy_model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate),
                                      loss_fun=NNL.NeuralNetworkLoss.crossentropy, loss_and_optimize=False)
            with self.graph.as_default():
                self.policy_model.target = tf.placeholder(shape=[None, ])
                self.policy_model.loss = self.policy_model.loss_fun(output=self.policy_model.output,
                                                                    target=self.policy_model.target,
                                                                    batch_size=1,
                                                                    axis=1)
                self.policy_model.loss = tf.reduce_mean(self.policy_model.loss * self.policy_model.action_state_value)
                if self.policy_model.update:
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        grads_and_vars = self.policy_model.optimizer.compute_gradients(self.policy_model.loss)
                        self.policy_model.train = self.policy_model.optimizer.apply_gradients(grads_and_vars)
                else:
                    grads_and_vars = self.policy_model.optimizer.compute_gradients(self.policy_model.loss)
                    self.policy_model.train = self.policy_model.optimizer.apply_gradients(grads_and_vars)

    def predict(self, state, epsilon=None):
        prob_weights = self.sess.run(fetches=self.policy_model.output, feed_dict={self.policy_model.input: state})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def _discount_and_norm_rewards(self):
        discounted_episode_rewards = np.zeros_like(self.episode_rewards)
        running_add = 0
        for t in reversed(range(len(self.episode_rewards))):
            running_add = running_add * self.gamma + self.episode_rewards[t]
            discounted_episode_rewards[t] = running_add

        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards
