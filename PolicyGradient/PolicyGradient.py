import tensorflow as tf
import numpy as np
import ReinforcementLearning.ReinforcementLearningModel as RLM
import NeuralNetwork.NeuralNetworkModel as NNM
import NeuralNetwork.NeuralNetworkUnit as NNU
import NeuralNetwork.NeuralNetworkLoss as NNL


class PolicyGradient(RLM.ReinforcementLearningModel):
    def __init__(self, env, gamma=0.8, batch_size=40, decay_rate=0.1, learning_rate=0.01, epsilon=0.05):
        super().__init__(env, gamma=gamma, decay_rate=decay_rate, learning_rate=learning_rate, epsilon=epsilon)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.batch_size = batch_size

    def fit(self):
        for i in range(self.env.episodes_size):
            state = self.env.Reset(iteration=i)
            while True:
                action = self.predict(state, self.epsilon)
                self.env.actions.append(action)
                new_state, reward, done = self.env.step()
                if done:
                    pass

    def _learn(self):
        pass

    def _construct_default_models(self):
        with self.graph.as_default():
            with tf.variable_scope('Policy_Network'):
                self.policy_model = NNM.NeuralNetworkModel(graph=self.graph)
                self.policy_model.build(NNU.NeuronLayer(hidden_dim=30, transfer_fun=tf.nn.sigmoid),
                                        input_dim=self.env.features_size)
                self.policy_model.build(NNU.NeuronLayer(hidden_dim=30, transfer_fun=None))
                self.policy_model.build(NNU.SoftMaxLayer())
            self.policy_model.mini_batch = self.batch_size
            self.policy_model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate),
                                      loss_fun=NNL.NeuralNetworkLoss.crossentropy)

    def predict(self, state, epsilon=None):
        prob_weights = self.sess.run(fetches=self.policy_model.output, feed_dict={})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action