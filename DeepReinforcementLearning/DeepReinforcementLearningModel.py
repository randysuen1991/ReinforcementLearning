import numpy as np
import random 
import copy
import sys

if 'C:\\Users\\randysuen\\Neural-Network' or  'C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Neural-Network' not in sys.path :
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')
    sys.path.append('C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Neural-Network')

import tensorflow as tf
import NeuralNetworkModel as NNM
import NeuralNetworkUnit as NNU
import NeuralNetworkLoss as NNL

import copy as cp
import warnings

if 'C:\\Users\\randysuen\\Reinforcement-Learning' or 'C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Reinforcement-Learning' not in sys.path :
    sys.path.append('C:\\Users\\randysuen\\Reinforcement-Learning')
    sys.path.append('C:\\Users\\ASUS\\Dropbox\\pycode\\mine\\Reinforcement-Learning')

import ReinforcementLearningModel as RLM



import matplotlib.pyplot as plt


class DeepQLearning(RLM.ReinforcementLearningModel):
    def __init__(self,env,memory_size, features_size, batch_size, episodes_size=10,
                 replace_target_size=100,learn_size=30, gamma=0.8,
                 decay_rate=0.1,learning_rate=0.01,epsilon=0.05,epsilon_increment=None,default=True):
        
        
        
        
        # Sometimes, the episodes size is determined by the  
        if hasattr(env,'episodes') :
            if len(env.episodes) < episodes_size :
                warnings.warn('The desired episodes size is less than the availebal episodes size, so set the \
                             episodes size to be the availebal size.')
                super().__init__(env=env, episodes_size = len(env.episodes), 
                      decay_rate=decay_rate, features_size = features_size, gamma=gamma,
                      learning_rate=learning_rate, epsilon=epsilon)
            else :
                super().__init__(env=env, episodes_size = episodes_size, 
                     decay_rate=decay_rate, features_size = features_size, gamma=gamma,
                      learning_rate=learning_rate, epsilon=epsilon)
        else :
            super().__init__(env=env, episodes_size = episodes_size, 
                     decay_rate=decay_rate, features_size = features_size, gamma=gamma,
                     learning_rate=learning_rate, epsilon=epsilon)
        
        
        
        
        self.replace_target_size = replace_target_size
        self.memory_size = memory_size
        self.learn_size = learn_size
        # the batch_size means how many data we would take from the memory.
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size, 2 * self.features_size + 2))
        # A deep Q model gets its own session.
        self.sess = tf.Session()
        self.cost_history = list()
        self.epsilon_max = 1 - self.epsilon
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        self.gamma = gamma
        
        
        if default :
            self._Construct_DefaultModels()
            
            self.eval_model.Compile(X_train_shape=(self.batch_size,self.features_size),optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),loss_fun=NNL.NeuralNetworkLoss.MeanSqaured)
            self.targ_model.Compile(X_train_shape=(self.batch_size,self.features_size),loss_and_optimize=False)
            
            self.sess.run(tf.global_variables_initializer())
            
            # If NOT using the default model, the following codes should be handled still.
            # Define replace_target_op here.
            
            
            eval_params = self._Get_parameters(self.eval_model)
            targ_params = self._Get_parameters(self.targ_model)
            
            assert len(eval_params) == len(targ_params)
            self.replace_target = [tf.assign(t, e) for t, e in zip(targ_params, eval_params)]
            
        # Should write codes to 
        else :
            pass
            
    def _Get_parameters(self,model):
#        print(model.layers)
        parameters_list = list()
        for layer in model.layers :
            parameters_dict = layer.parameters
#            print(parameters_dict)
            for _, parameters in parameters_dict.items():
                parameters_list.append(parameters)
                
        return parameters_list 
    
    
    
    def _Construct_DefaultModels(self):
        
        self.eval_model = NNM.NeuralNetworkModel()
#        self.eval_model.Build(NNU.NeuronLayer(hidden_dim=15))
        self.eval_model.Build(NNU.NeuronLayer(hidden_dim=10))
        self.eval_model.Build(NNU.NeuronLayer(hidden_dim=self.actions_size))
        
        # target model and eval model share the same structure.
        self.targ_model = NNM.NeuralNetworkModel()
#        self.targ_model.Build(NNU.NeuronLayer(hidden_dim=15))
        self.targ_model.Build(NNU.NeuronLayer(hidden_dim=10))
        self.targ_model.Build(NNU.NeuronLayer(hidden_dim=self.actions_size))
        
    
    def Fit(self):
        
        for i in range(self.episodes_size) :
            
            step = 0
            state = self.env.Reset()
            
            while True :
                
                
                
                action = self.Predict(state)
                
                action = self.env.DealAction(action)
                
                if action == -1 :
                    print('Sell')
                elif action == 0 :
                    print('Hold')
                elif action == 1 :
                    print('Buy')
                
                
                new_state, reward, done = self.env.Step(action)
                
                print('Reward:',reward)
                
                self._Store_Transition(state.ravel(),action,reward,new_state.ravel())
            
                if (step > self.learn_size) and (step % 5 == 0) :
                    self._Learn()
                
                
                
                
                
                state = new_state
                step += 1
            
                if step % self.replace_target_size == 0:
                    self.sess.run(self.replace_target)
                    print('\ntarget_params_replaced\n')
            
                if done :
                    print('Training over!')
                    plt.plot(self.cost_history)
                    break
            
            
                
    # If an epsilon is passed, it would be greedy strategy, and vice versa.
    def Predict(self, state, epsilon=None) :
        
        actions = self.eval_model.Predict(X_test=state)
        print(actions)
        action = np.argmax(actions)
        
        try :
            if np.random.uniform(0,1) < 1 - epsilon + epsilon / len(self.actions) :
                return action
            else :
                actions_list = copy.copy(self.actions)
                actions_list.remove(action)
                return random.choice(actions_list)
        except :
            return action
    
    def _Store_Transition(self,state,action,reward,new_state):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state,action,reward,new_state))
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1
        
    
    
    def _Learn(self):
        
        if self.memory_counter > self.memory_size :
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else :
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
            
        batch_memory = self.memory[sample_index,:]
        
        
        
        q_next, q_eval = self.sess.run([self.targ_model.output, self.eval_model.output],feed_dict={self.targ_model.input:batch_memory[:,-self.features_size:],
                                       self.eval_model.input:batch_memory[:,:self.features_size]})
        
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # pick the actions done in those steps.
        eval_act_index = batch_memory[:, self.features_size].astype(int)
        # get the reward if taking that action.
        reward = batch_memory[:, self.features_size + 1]
        
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        
        _, self.cost = self.sess.run([self.eval_model.train, self.eval_model.loss],
                                     feed_dict={self.eval_model.input: batch_memory[:, :self.features_size],
                                                self.eval_model.target: q_target})
        
        self.cost_history.append(self.cost)
        
        
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        
        
class DoubleDeepQLearning(DeepQLearning):
    def __init__(self,states,actions,env,episodes_size,
                 features_size, memory_size, batch_size,replace_target_size=300,learn_size=150,
                 decay_rate=0.1,learning_rate=0.01,epsilon=0.05,epsilon_increment=None,default=True):
        
        super().__init__(states,actions,env,episodes_size,
                 features_size, memory_size, batch_size,replace_target_size=300,learn_size=150,
                 decay_rate=0.1,learning_rate=0.01,epsilon=0.05,epsilon_increment=None,default=True)
    
    def _Learn(self):
        if self.memory_counter > self.memory_size :
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else :
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
            
        batch_memory = self.memory[sample_index,:]
        
        self.eval_model.Compile(X_train_shape=batch_memory[:,:self.n_features].shape,optimizer=tf.train.RMSPropOptimizer,loss_fun=NNL.NeuralNetworkLoss.MeanSqaured)
        self.targ_model.Compile(X_train_shape=batch_memory[:,-self.n_features:].shape,loss_and_optimize=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # pick the actions done in those steps.
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # get the reward if taking that action.
        reward = batch_memory[:, self.n_features + 1]
        
        
        q_next, q_eval4next = self.sess.run([self.targ_model.output, self.eval_model.output],
                                            feed_dict = {self.targ_model.input:batch_memory[:,-self.n_features:],
                                                         self.eval_model.input:batch_memory[:,-self.n_features:]})
        
        q_eval = self.sess.run(self.eval_model.output, 
                               feed_dict = {self.eval_model.input : batch_memory[:, :self.n_features]})    
    
        q_target = q_eval.copy()
        
        max_a4next = np.argmax(q_eval4next, axis=1)
        selected_q_next = q_next[batch_index, max_a4next] 
        
        
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        
        _, self.cost = self.sess.run([self.eval_model.train, self.eval_model.loss],
                                     feed_dict={self.eval_model.input : batch_memory[:, :self.n_features],
                                                self.eval_model.target : q_target})
        
        
        self.cost_history.append(self.cost)
        
        
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        
class DeeepQLearningPrioReply(DeepQLearning):
    def __init__(self,states,actions,env,episodes_size,
                 features_size, memory_size, batch_size,replace_target_size=300,learn_size=150,
                 decay_rate=0.1,learning_rate=0.01,epsilon=0.05,epsilon_increment=None,default=True):
        super().__init__(states,actions,env,episodes_size,
                 features_size, memory_size, batch_size,replace_target_size=300,learn_size=150,
                 decay_rate=0.1,learning_rate=0.01,epsilon=0.05,epsilon_increment=None,default=True)
    
    def _Construct_DefaultModels(self):
        
        self.eval_model = NNM.NeuralNetworkModel()
        self.eval_model.Build(NNU.NeuronLayer(hidden_dim=20))
        self.eval_model.Build(NNU.NeuronLayer(hidden_dim=15))
        self.eval_model.Build(NNU.NeuronLayer(hidden_dim=self.actions_size))
        
        # For priority reply
        self.eval_model.ISWeights = tf.placeholder(tf.float32, [None, 1])
        
        # target model and eval model share the same structure.
        self.targ_model = cp.deepcopy(self.eval_model)
    
    
    def _Store_Transition(self,state,action,reward,new_state):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state,action,reward,new_state))
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1
     
        
""" Be careful this is a diamond inheritnace. """
class DoubleDeepQLearningPrioReply(DoubleDeepQLearning,DeeepQLearningPrioReply):
    def __init__(self,states,actions,env,episodes_size,
                 features_size, memory_size, batch_size,replace_target_size=300,learn_size=150,
                 decay_rate=0.1,learning_rate=0.01,epsilon=0.05,epsilon_increment=None,default=True):
        super().__init__(states,actions,env,episodes_size,
                 features_size, memory_size, batch_size,replace_target_size=300,learn_size=150,
                 decay_rate=0.1,learning_rate=0.01,epsilon=0.05,epsilon_increment=None,default=True)
        