# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:25:08 2021

@author: Millend
"""

# Training the AI
import torch
import torch.nn.functional as F
from env import Enviro
from model import ActorCritic
from torch.autograd import Variable
import numpy as np
import time

# Implementing a function to make sure the models share the same gradient
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank) # shifting the seed with rank to asynchronize each training agent
    env = Enviro() # creating an optimized environment thanks to the create_atari_env function
    env.seed(params.seed + rank) # aligning the seed of the environment on the seed of the agent
    model = ActorCritic(env.observation_space, env.action_space) # creating the model from the ActorCritic class
    state,iterationx = env.train_reset(params.seed + rank) # state is a numpy array of size 1*42*42, in black & white
    iteration=iterationx
    state = torch.from_numpy(state).type(torch.float) # converting the numpy array into a torch tensor
    done = True # when the game is done
    episode_length = 0 #initializing the length of an episode to 0
    reward_sum = 0 # initializing the sum of rewards to 0
    start_time = time.time() # getting the starting time to measure the computation time
    while (episode_length<=params.max_episode_length): #repeat           
        model.load_state_dict(shared_model.state_dict()) #synchronizing with the shared model - the agent gets the shared model to do an exploration on num_steps
        values = [] #initializing the list of values (V(S))
        log_probs = [] #initializing the list of log probabilities
        rewards = [] #initializing the list of rewards
        entropies = [] #initializing the list of entropies
        for step in range(params.num_steps): #going through the num_steps exploration steps
            value, action_values = model(Variable(state.unsqueeze(0))) # getting from the model the output V(S) of the critic, the output Q(S,A) of the actor
            prob = F.softmax(action_values) # generating a distribution of probabilities of the Q-values according to the softmax: prob(a) = exp(prob(a))/sum_b(exp(prob(b)))
            log_prob = F.log_softmax(action_values) # generating a distribution of log probabilities of the Q-values according to the log softmax: log_prob(a) = log(prob(a))
            entropy = -(log_prob * prob).sum(1) # H(p) = - sum_x p(x).log(p(x))
            entropies.append(entropy) # storing the computed entropy
            action = prob.multinomial(1).data # selecting an action by taking a random draw from the prob distribution
            log_prob = log_prob.gather(1, Variable(action)) # getting the log prob associated to this selected action
            values.append(value) # storing the value V(S) of the state
            log_probs.append(log_prob) # storing the log prob of the action
            state, reward, done, _ = env.train_step(action.numpy(),iteration, iterationx) # playing the selected action, reaching the new state, and getting the new reward
            iteration=(iteration+1)%(env.X_train.shape[0])
            #print(reward)
            reward_sum=reward_sum+reward
            if done: # if the episode is done:
                print("Rank {}, Time {}, episode reward {}, episode length {}".format(rank, time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
                episode_length += 1
                reward_sum = 0 # reinitializing the sum of rewards
                state,iterationx = env.train_reset(params.seed + rank) # we restart the environment
                iteration=iterationx
            state = torch.from_numpy(state).type(torch.float) # tensorizing the new state
            rewards.append(reward) # storing the new observed reward
            if done: # if we are done
                break # we stop the exploration and we directly move on to the next step: the update of the shared model
        R = torch.zeros(1, 1) # intializing the cumulative reward
        if not done: # if we are not done:
            value, _ = model(Variable(state.unsqueeze(0))) # we initialize the cumulative reward with the value of the last shared state
            R = value.data # we initialize the cumulative reward with the value of the last shared state
        values.append(Variable(R)) # storing the value V(S) of the last reached state S
        policy_loss = 0 # initializing the policy loss
        value_loss = 0 # initializing the value loss
        R = Variable(R) # making sure the cumulative reward R is a torch Variable
        gae = torch.zeros(1, 1) # initializing the Generalized Advantage Estimation to 0
        #print(rewards)
        for i in reversed(range(len(rewards))): # starting from the last exploration step and going back in time
            R = params.gamma * R + rewards[i] # R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
            advantage = R - values[i] # R is an estimator of Q at time t = i so advantage_i = Q_i - V(state_i) = R - value[i]
            value_loss = value_loss + 0.5 * advantage.pow(2) # computing the value loss
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data # computing the temporal difference
            gae = gae * params.gamma * params.tau + TD # gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i] # computing the policy loss
        #print(episode_length, iteration, R.data, advantage.data, policy_loss.data , value_loss.data)
        optimizer.zero_grad() # initializing the optimizer
        (policy_loss + value_loss).backward() # we give 2x more importance to the policy loss than the value loss because the policy loss is smaller
        torch.nn.utils.clip_grad_norm(model.parameters(), 40) # clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
        ensure_shared_grads(model, shared_model) # making sure the model of the agent and the shared model share the same gradient
        optimizer.step() # running the optimization step
