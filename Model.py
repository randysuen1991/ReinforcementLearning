import numpy as np
import random 
import copy
import sys
import pandas as pd

if 'C:\\Users\\randysuen\\Neural-Network' or 'C:\\Users\\randysuen\\Neural-Network' not in sys.path :
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')
    sys.path.append('C:\\Users\\randysuen\\Neural-Network')

import tensorflow as tf
import NeuralNetworkModel as NNM
import NeuralNetworkUnit as NNU
import NeuralNetworkLoss as NNL

import copy as cp


"""
This file consists three parts:
    The first is the action-value function, Q, and state-value function, V, for policy pi.
    The second one is the learning algorithm, include SARSA, Q-learning, .... .
    The last is the model itself which uses the learning algorith to make decision.
    
"""

# This class should be a double keys one value dictionary.
class QTable():
    def __init__(self,states,actions):
        self.QTable = pd.DataFrame(index=states,columns=actions)        




class ReinforcementLearningModel():
    def __init__(self,states,actions,env,episodes_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05):
        self.states = states
        self.actions = actions
        self.env = env
        self.episodes_size = episodes_size
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.Q = Q_Generator(states,actions)
        
    def Predict(self):
        raise NotImplementedError
    
    def Fit(self):
        raise NotImplementedError
    
    
class QLearning(ReinforcementLearningModel):
    def __init__(self,states,actions,env,episodes_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05):
        super().__init__(states,actions,env,episodes_size,decay_rate,learning_rate,epsilon)
        
        
    def Fit(self):
        for _ in range(self.episodes_size):
            done = False
            state = random.choice(self.states)
            while True :
                if done :
                    break
                action = self.Predict(state,self.epsilon)
                new_state, r, done, _  = self.env.Step(action)
                new_action = self.Predict(state)
                self.Q[state,action] = self.Q[state,action] + self.learning_rate * (r + self.decay_rate * self.Q[new_state,new_action] - self.Q[state,action])
                state = new_state
                
    
    def Predict(self, state, epsilon=None):
        action = max(self.Q[state],key=self.Q[state].get)
        try :
            if np.random.uniform(0,1) < 1 - epsilon + epsilon / len(self.actions) :
                return action
            else :
                actions_list = copy.copy(self.actions)
                actions_list.remove(action)
                return random.choice(actions_list)
        except :
            return action
    
class Sarsa(ReinforcementLearningModel):
    def __init__(self,states,actions,env,episodes_size,decay_rate=0.1,learning_rate=0.01,epsilon=0.05):
        super().__init__(states,actions,env,episodes_size,decay_rate,learning_rate,epsilon)
    
class DeepQLearning(ReinforcementLearningModel):
    def __init__(self,states,actions,env,episodes_size,replace_target_size,
                 memory_size, batch_size,
                 decay_rate=0.1,learning_rate=0.01,epsilon=0.05,default=True):
        super().__init__(states,actions,env,episodes_size,decay_rate,learning_rate,epsilon)
        self.replace_target_size = replace_target_size
        self.memory_size
        # the batch_size means how many data we would take from the memory.
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size,))
        if default :
            self._Construct_DefaultModels()
    
    def _Construct_DefaultModels(self):
        
        
        
        self.eval_model = NNM.NeuralNetworkModel()
        self.eval_model.Build(NNU.NeuronLayer(hidden_dim=len(self.actions)))
        self.eval_model.Build(NNU.NeuronLayer(hidden_dim=))
        self.eval_model.Build(NNU.NeuronLayer(hidden_dim=))
        
        # target model and eval model share the same structure.
        self.targ_model = cp.deepcopy(self.eval_model)
        
        
        
        
    
    def Fit(self):
        step = 0
        state = self.env.Reset()
        
        while True :
            action = self.Predict(state)
            state_, reward, done = self.env.Step(action)
            if (step > 200) and (step % 5 == 0) :
                self.model.
            state = state_
            step += 1
            
    def Predict(self, action, epsilon=None) :
        pass
    
    def _Store_Transition(self,state,action,reward,new_state):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state,action,reward,new_state))
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1
        
    def _Replace_Target_Params(self):
        pass
    
    def _Learn(self):
        if self.memory_counter > self.memory_size :
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else :
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory = self.memory[sample_index,:]
        
        new_q = self.targ_model.sess.run()
        eval_q = self.eval_model.sess.run()
        
        
    