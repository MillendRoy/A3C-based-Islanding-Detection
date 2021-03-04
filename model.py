# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:11:20 2021

@author: Millend
"""
# Importing the librairies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= (std / torch.sqrt(out.pow(2).sum(1).expand_as(out.t()))).t() # thanks to this initialization, we have var(out) = std^2
    return out

# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m):
  classname=m.__class__.__name__
  if classname.find('Linear')!=-1:     
    weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
    fan_in = weight_shape[1] # dim1
    fan_out = weight_shape[0] # dim0
    w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
    m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
    m.bias.data.fill_(0) # initializing all the bias with zeros

# Making the A3C brain
class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(num_inputs,16)
        num_outputs = action_space # getting the number of possible actions
        self.critic_linear = nn.Linear(16, 1) # full connection of the critic: output = V(S)
        self.actor_linear = nn.Linear(16, num_outputs) # full connection of the actor: output = Q(S,A)
        self.apply(weights_init) # initilizing the weights of the model with random weights
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01) # setting the standard deviation of the actor tensor of weights to 0.01
        self.actor_linear.bias.data.fill_(0) # initializing the actor bias with zeros
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0) # setting the standard deviation of the critic tensor of weights to 0.01
        self.critic_linear.bias.data.fill_(0) # initializing the critic bias with zeros
        self.train() # setting the module in "train" mode to activate the dropouts and batchnorms

    def forward(self, inputs):
        x = F.elu(self.fc(inputs)) 
        return self.critic_linear(x), self.actor_linear(x)
