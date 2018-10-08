import numpy as np
import random 
import copy as cp
import tensorflow as tf
from NeuralNetwork import NeuralNetworkModel as NNM
from NeuralNetwork import NeuralNetworkUnit as NNU
from NeuralNetwork import NeuralNetworkLoss as NNL
from ReinforcementLearning import ReinforcementLearningModel as RLM
import matplotlib.pyplot as plt


class DeepQLearning(RLM.ReinforcementLearningModel):
    def __init__(self, env, memory_size=50, batch_size=40,
                 replace_target_size=100, learn_size=30, gamma=0.8,
                 decay_rate=0.1, learning_rate=0.1, epsilon=0.5, default=True):
        super().__init__(env=env, gamma=gamma, decay_rate=decay_rate, learning_rate=learning_rate, epsilon=epsilon)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.replace_target_size = replace_target_size
        self.memory_size = memory_size
        self.learn_size = learn_size
        # the batch_size means how many data we would take from the memory.
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size, 2 * self.env.features_size + 2))
        self.cost_history = list()
        self.epsilon = epsilon
        self.gamma = gamma
        self.eval_model = None
        self.targ_model = None
        assert self.memory_size > self.batch_size

        if default:
            self._construct_default_models()
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
            # If NOT using the default model, the following codes should be handled still.
            # Define replace_target_op here.
            eval_params = self._get_parameters(self.eval_model.NNTree.root)
            targ_params = self._get_parameters(self.targ_model.NNTree.root)
            assert len(eval_params) == len(targ_params)
            with self.graph.as_default():
                self.replace_target = [tf.assign(t, e) for t, e in zip(targ_params, eval_params)]
            
        # Should write codes here.
        else:
            pass
            
    def _get_parameters(self, layer):
        parameters_list = list()
        parameters_dict = layer.parameters
        for _, parameters in parameters_dict.items():
            parameters_list.append(parameters)
        for _, son in layer.sons.items():
            son_parameters_list = self._get_parameters(son)
            parameters_list += son_parameters_list
        return parameters_list 

    def _construct_default_models(self):
        with self.graph.as_default():
            with tf.variable_scope('eval'):
                self.eval_model = NNM.NeuralNetworkModel(graph=self.graph)

                self.eval_model.build(NNU.NeuronLayer(hidden_dim=30, transfer_fun=tf.nn.sigmoid),
                                      input_dim=self.env.features_size)
                self.eval_model.build(NNU.BatchNormalization())
                self.eval_model.build(NNU.NeuronLayer(hidden_dim=20, transfer_fun=tf.nn.sigmoid))
                self.eval_model.build(NNU.BatchNormalization())
                self.eval_model.build(NNU.NeuronLayer(hidden_dim=self.actions_size))
                # self.eval_model.Build(NNU.BatchNormalization())

            self.eval_model.batch_size = self.batch_size
            self.eval_model.mini_batch = self.eval_model.batch_size
            self.eval_model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate),
                                    loss_fun=NNL.NeuralNetworkLoss.meansquared)
            self.eval_model.sess.close()
            # target model and eval model share the same structure.
            with tf.variable_scope('target'):
                self.targ_model = NNM.NeuralNetworkModel(graph=self.graph)
                self.targ_model.build(NNU.NeuronLayer(hidden_dim=30, transfer_fun=tf.nn.sigmoid),
                                      input_dim=self.env.features_size)
                self.targ_model.build(NNU.BatchNormalization())
                self.targ_model.build(NNU.NeuronLayer(hidden_dim=20, transfer_fun=tf.nn.sigmoid))
                self.targ_model.build(NNU.BatchNormalization())
                self.targ_model.build(NNU.NeuronLayer(hidden_dim=self.actions_size, transfer_fun=tf.nn.sigmoid))
                # self.targ_model.build(NNU.BatchNormalization())
            self.targ_model.sess.close()
    
    def fit(self, plot_cost=False):
        for i in range(self.env.episodes_size):
            step = 0
            state = self.env.Reset(iteration=i)
            while True:
                action = self.predict(state, self.epsilon)
                self.env.actions.append(action)
                new_state, reward, done = self.env.Step()
                self._store_transition(state.ravel(), action, reward, new_state.ravel())
                if done:
                    action = self.predict(new_state, self.epsilon)
                    self.env.actions.append(action)
                    break
                if (step > self.learn_size) and (step % 5 == 0):
                    self._learn()
                state = new_state
                step += 1
                if step % self.replace_target_size == 0:
                    self.sess.run(self.replace_target)
        if plot_cost:
            plt.plot(self.cost_history)

        print('Training over!')
                
    # If an epsilon is passed, it would be greedy strategy, and vice versa.
    def predict(self, state, epsilon=None):
        actions = self.sess.run(fetches=self.eval_model.output,
                                feed_dict={self.eval_model.input: state,
                                           self.eval_model.on_train: False})

        # if the environment has specific action action format.
        action = self.env.dealaction(np.argmax(actions))
        
        try:
            if np.random.uniform(0, 1) < (1 - epsilon + epsilon / len(self.env.actions_space)):
                return action
            else:
                actions_list = cp.copy(self.env.actions_space)
                actions_list.remove(action)
                return random.choice(actions_list)
        except TypeError:
            return action
    
    def _store_transition(self, state, action, reward, new_state):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state, action, reward, new_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def _learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            
        batch_memory = self.memory[sample_index, :]
        q_next, q_eval = self.sess.run([self.targ_model.output, self.eval_model.output],
                                       feed_dict={self.targ_model.input: batch_memory[:, -self.env.features_size:],
                                                  self.eval_model.input: batch_memory[:, :self.env.features_size],
                                                  self.targ_model.on_train: False,
                                                  self.eval_model.on_train: False})
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # pick the actions done in those steps.
        eval_act_index = batch_memory[:, self.env.features_size].astype(int)
        actions_index = list()
        for action in eval_act_index:
            actions_index.append(self.env.DealAction_Inverse(action))
        # get the reward if taking that action.
        reward = batch_memory[:, self.env.features_size + 1]
        q_target[batch_index, actions_index] = reward + self.gamma * np.max(q_next, axis=1)
        _, cost = self.sess.run([self.eval_model.train, self.eval_model.loss],
                                feed_dict={self.eval_model.input: batch_memory[:, :self.env.features_size],
                                           self.eval_model.target: q_target,
                                           self.eval_model.on_train: True})
        self.cost_history.append(cost)
    
    def backtest(self, states):
        actions = list()
        for state in states:
            action = self.predict(state)
            actions.append(action)
        return actions

    def print_output_detail(self, states):
        for state in states:
            self.eval_model.print_output_detail(state, sess=self.sess)


class DoubleDeepQLearning(DeepQLearning):
    def __init__(self, env, memory_size=50, batch_size=40,
                 replace_target_size=100, learn_size=30, gamma=0.8,
                 decay_rate=0.1, learning_rate=0.1, epsilon=0.5, default=True):
        
        super().__init__(env=env, memory_size=memory_size, batch_size=batch_size,
                         replace_target_size=replace_target_size, learn_size=learn_size, gamma=gamma,
                         decay_rate=decay_rate, learning_rate=learning_rate, epsilon=epsilon, default=default)
    
    def _learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # pick the actions done in those steps.
        eval_act_index = batch_memory[:, self.env.features_size].astype(int)
        actions_index = list()
        for action in eval_act_index:
            actions_index.append(self.env.DealAction_Inverse(action))
        # get the reward if taking that action.
        reward = batch_memory[:, self.env.features_size + 1]
        q_next, q_eval4next = self.sess.run([self.targ_model.output, self.eval_model.output],
                                            feed_dict={self.targ_model.input: batch_memory[:, -self.env.features_size:],
                                                       self.eval_model.input: batch_memory[:, -self.env.features_size:],
                                                       self.targ_model.on_train: False,
                                                       self.eval_model.on_train: False
                                                       })
        q_eval = self.sess.run(self.eval_model.output, 
                               feed_dict={self.eval_model.input: batch_memory[:, :self.env.features_size],
                                          self.eval_model.on_train: False})
        q_target = q_eval.copy()
        max_a4next = np.argmax(q_eval4next, axis=1)

        selected_q_next = q_next[batch_index, max_a4next]
        q_target[batch_index, actions_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self.eval_model.train, self.eval_model.loss],
                                     feed_dict={self.eval_model.input: batch_memory[:, :self.env.features_size],
                                                self.eval_model.target: q_target,
                                                self.eval_model.on_train: True})

        self.cost_history.append(self.cost)


class DeepQLearningPrioReply(DeepQLearning):
    def __init__(self, env, memory_size=50, batch_size=40, replace_target_size=100,
                 learn_size=30, gamma=0.8, decay_rate=0.1, learning_rate=0.1, epsilon=0.5, default=True):

        super().__init__(env, memory_size, batch_size, replace_target_size, learn_size, gamma,
                         decay_rate, learning_rate, epsilon, default)
    
    def _construct_default_models(self):
        
        self.eval_model = NNM.NeuralNetworkModel()
        self.eval_model.build(NNU.NeuronLayer(hidden_dim=20))
        self.eval_model.build(NNU.NeuronLayer(hidden_dim=15))
        self.eval_model.build(NNU.NeuronLayer(hidden_dim=self.actions_size))
        
        # For priority reply
        self.eval_model.ISWeights = tf.placeholder(tf.float32, [None, 1])
        
        # target model and eval model share the same structure.
        self.targ_model = cp.deepcopy(self.eval_model)

    def _store_transition(self, state, action, reward, new_state):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state, action, reward, new_state))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
     

class DoubleDeepQLearningPrioReply(DoubleDeepQLearning, DeepQLearningPrioReply):
    def __init__(self, env, memory_size=50, batch_size=40, replace_target_size=100,
                 learn_size=30, gamma=0.8, decay_rate=0.1, learning_rate=0.1, epsilon=0.5, default=True):
        super().__init__(env, memory_size, batch_size, replace_target_size, learn_size, gamma,
                         decay_rate, learning_rate, epsilon, default)

        
class DuelingDeepQLearning(DeepQLearning):
    def __init__(self, env, memory_size=50, batch_size=40, replace_target_size=100,
                 learn_size=30, gamma=0.8, decay_rate=0.1, learning_rate=0.1, epsilon=0.5, default=True):
        super().__init__(env, memory_size, batch_size, replace_target_size, learn_size, gamma,
                         decay_rate, learning_rate, epsilon, default)

    def _construct_default_models(self):
        with self.graph.as_default():
            with tf.variable_scope('eval'):
                self.eval_model = NNM.NeuralNetworkModel(graph=self.graph)
                self.eval_model.build(NNU.NeuronLayer(hidden_dim=10, transfer_fun=tf.nn.sigmoid),
                                      input_dim=self.env.features_size)
                self.eval_model.build(NNU.BatchNormalization())
                # self.eval_model.build(NNU.NeuronLayer(hidden_dim=5, transfer_fun=tf.nn.sigmoid))
                # self.eval_model.build(NNU.BatchNormalization())
                self.eval_model.split(names=['adv', 'value'])
                self.eval_model.build(NNU.NeuronLayer(hidden_dim=1), name='value')
                self.eval_model.build(NNU.NeuronLayer(hidden_dim=self.actions_size), name='adv')
                self.eval_model.build(NNU.ReduceMean(), name='adv')
                self.eval_model.merge(op='add', names=['adv', 'value'])

            self.eval_model.batch_size = self.batch_size
            self.eval_model.mini_batch = self.eval_model.batch_size
            self.eval_model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate),
                                    loss_fun=NNL.NeuralNetworkLoss.meansquared)
            self.eval_model.sess.close()

            with tf.variable_scope('target'):
                self.targ_model = NNM.NeuralNetworkModel(graph=self.graph)
                self.targ_model.build(NNU.NeuronLayer(hidden_dim=10, transfer_fun=tf.nn.sigmoid),
                                      input_dim=self.env.features_size)
                self.targ_model.build(NNU.BatchNormalization())
                # self.targ_model.build(NNU.NeuronLayer(hidden_dim=5, transfer_fun=tf.nn.sigmoid))
                # self.targ_model.build(NNU.BatchNormalization())
                self.targ_model.split(names=['adv', 'value'])
                self.targ_model.build(NNU.NeuronLayer(hidden_dim=1), name='value')
                self.targ_model.build(NNU.NeuronLayer(hidden_dim=self.actions_size), name='adv')
                self.targ_model.build(NNU.ReduceMean(), name='adv')
                self.targ_model.merge(op='add', names=['adv', 'value'], output_name='last')

            self.targ_model.sess.close()

