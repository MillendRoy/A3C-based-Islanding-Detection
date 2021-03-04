# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:16:07 2021

@author: Millend
"""

# Test Agent
import torch
import torch.nn.functional as F
from env import Enviro
from model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque
import numpy as np

# Making the test agent (won't update the model but will just use the shared model to explore)
def testt(rank, params, shared_model):
    torch.manual_seed(params.seed + rank) # asynchronizing the test agent
    env = Enviro() # running an environment with a video
    env.seed(params.seed + rank) # asynchronizing the environment
    model = ActorCritic(env.observation_space, env.action_space) # creating one model
    model.eval() # putting the model in "eval" model because it won't be trained
    state,iterationx = env.test_reset(params.seed + rank) # state is a numpy array of size 1*42*42, in black & white
    iteration=iterationx# getting the input images as numpy arrays
    state = torch.from_numpy(state).type(torch.float) # converting them into torch tensors
    reward_sum = 0 # initializing the sum of rewards to 0
    done = True # initializing done to True
    start_time = time.time() # getting the starting time to measure the computation time
    actions = deque(maxlen=100) # cf https://pymotw.com/2/collections/deque.html
    episode_length = 0 # initializing the episode length to 0
    while (episode_length<100): # repeat
        model.load_state_dict(shared_model.state_dict())
        value, action_value = model(Variable(state.unsqueeze(0), requires_grad=False))
        prob = F.softmax(action_value)
        action = prob.max(1)[1].data.numpy() # the test agent does not explore, it directly plays the best action
        state, reward, done, _ = env.test_step(action[0],iteration,iterationx) # done = done or episode_length >= params.max_episode_length
        iteration=(iteration+1)%(env.X_test.shape[0])
        reward_sum += reward
        if done: # printing the results at the end of each part
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            episode_length += 1 # incrementing the episode length by one
            reward_sum = 0 # reinitializing the sum of rewards
            actions.clear() # reinitializing the actions
            state,iterationx = env.test_reset(params.seed + rank) # we restart the environment
            iteration=iterationx # reinitializing the environment
            time.sleep(10) # doing a one minute break to let the other agents practice (if the game is done)
        state = torch.from_numpy(state).type(torch.float) # new state and we continue
        