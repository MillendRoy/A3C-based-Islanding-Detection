# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:01:04 2021

@author: Millend
"""

# Main code
from __future__ import print_function
import os
import torch
import torch.multiprocessing as mp
from model import ActorCritic
from train import train
from testt import testt
import my_optim
from env import Enviro


# Gathering all the parameters (that we can modify to explore)
class Params():
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1.
        self.seed = 1
        self.num_processes = 16
        self.num_steps = 1
        self.max_episode_length = 100

# Main run
os.environ['OMP_NUM_THREADS'] = '1' # 1 thread per core
params = Params() # creating the params object from the Params class, that sets all the model parameters
torch.manual_seed(params.seed) # setting the seed (not essential)
env=Enviro()
shared_model = ActorCritic(env.observation_space, env.action_space) # shared_model is the model shared by the different agents (different threads in different cores)
shared_model.share_memory() # storing the model in the shared memory of the computer, which allows the threads to have access to this shared memory even if they are in different cores
optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=params.lr) # the optimizer is also shared because it acts on the shared model
optimizer.share_memory() # same, we store the optimizer in the shared memory so that all the agents can have access to this shared memory to optimize the model
testt(params.num_processes, params, shared_model)
'''processes = [] # initializing the processes with an empty list
p = mp.Process(target=testt, args=(params.num_processes, params, shared_model)) # allowing to create the 'test' process with some arguments 'args' passed to the 'test' target function - the 'test' process doesn't update the shared model but uses it on a part of it - torch.multiprocessing.Process runs a function in an independent thread
p.start() # starting the created process p
processes.append(p) # adding the created process p to the list of processes
'''
for rank in range(0, params.num_processes): # making a loop to run all the other processes that will be trained by updating the shared model
    train(rank, params, shared_model, optimizer)    
    '''p = mp.Process(target=train, args=(rank, params, shared_model, optimizer))
    p.start()
    processes.append(p)'''
'''for p in processes: # creating a pointer that will allow to kill all the threads when at least one of the threads, or main.py will be killed, allowing to stop the program safely
    p.join()
'''