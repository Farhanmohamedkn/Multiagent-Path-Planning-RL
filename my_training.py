#!/usr/bin/env python
# coding: utf-8

# # Pathfinding via Reinforcement and Imitation Multi-Agent Learning (PRIMAL)
# 
# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
# `tensorboard --logdir train_primal`

# In[3]:


#this should be the thing, right?
from __future__ import division

import gym
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import threading
import time
import scipy.signal as signal
import os
import GroupLock
import multiprocessing

import mapf_gym as mapf_gym
import pickle
import imageio

from ACNet_pytorch_my_change import ACNet

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


# Define constants
GAMMA = 0.99  # Discount factor
LR = 0.001    # Learning rate
BATCH_SIZE = 32
NUM_EPISODES = 10
MAX_STEPS = 250
DIAG_MVMT = False  # Allow diagonal movements
ENVIRONMENT_SIZE = (10,20)  # Size of the grid environment
GRID_SIZE = 10
OBSTACLE_DENSITY = (0,.3)
FULL_HELP = False
LR=1e-2
RNN_SIZE = 512
EXPERIENCE_BUFFER_SIZE=5

model_path="/home/noushad/Master_thesis/my_PRIMAL/pytorch_model/state_dict_model.pt"




# Set up the model, optimizer, and other parameters
a_size = 5 + int(DIAG_MVMT)*4 # Define action size #action_size=5 #action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST


## Initialize Gym environment
num_agents=1 #number of agents

gameEnv = mapf_gym.MAPFEnv(num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE,
                           observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)





# initialize model,optimizer,and loss function
model= ACNet('global', a_size, None, True, GRID_SIZE, 'global')
optimizer = torch.optim.NAdam(model.parameters(), lr=LR)
# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

agentID=1
episode_inv_count=0

# Training 
TRAINING=True

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


for episode in range(NUM_EPISODES):
        model.train()
       
        state = gameEnv._reset(agentID)
        print("ENV_reseted")
        c_in = torch.zeros(1, RNN_SIZE)
        h_in = torch.zeros(1, RNN_SIZE)
        rnn_state             =(c_in, h_in)
        rnn_state0            = rnn_state
        
        global rollouts
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            
            episode_buffer, episode_values = [], []
            d = False
            validActions          = gameEnv._listNextValidActions(agentID)
            s                     = gameEnv._observe(agentID)
            blocking              = False
            p=gameEnv.world.getPos(agentID)
            # print(p)
            
            on_goal               = gameEnv.world.goals[p[0],p[1]]==agentID

        
            s                     = gameEnv._observe(agentID)
            
            RewardNb = 0 
            wrong_blocking  = 0
            wrong_on_goal=0
            inputs=s[0] #pso,goalobs maps
            inputs=np.array(inputs)
            inputs=torch.tensor(inputs, dtype=torch.float32)
            # Convert to PyTorch tensor of size 4x10x10
            # inputs=torch.tensor(inputs, dtype=torch.float32)

            goal_pos=s[1] #dx,dy,mag
            goal_pos=np.array(goal_pos)
            goal_pos=torch.tensor(goal_pos, dtype=torch.float32)
            # Convert to PyTorch tensor of size 2x1
            # goal_pos=torch.tensor(goal_pos, dtype=torch.float32)
            # Set model to evaluation mode
            model.eval()
            with torch.no_grad():
                a_dist, v, state_in,state_out,pred_blocking,pred_on_goal,policy_sig=model(inputs=inputs,goal_pos=goal_pos,state_init=rnn_state0,training=True)
            # rnn_state             = state_in #it should be state innit
            rnn_state0            = state_out
            


            # Check if the argmax of a_dist is in validActions
            if not (torch.argmax(a_dist.flatten()).item() in validActions):
                episode_inv_count += 1

            # Initialize train_valid with zeros and set valid actions
            train_valid = torch.zeros(a_size)
            train_valid[validActions] = 1

            # Get the valid action distribution and normalize it
            valid_dist = a_dist[0, validActions]
            valid_dist /= torch.sum(valid_dist)

            if TRAINING:
                if (pred_blocking.flatten()[0] < 0.5) == blocking:
                    wrong_blocking += 1
                if (pred_on_goal.flatten()[0] < 0.5) == on_goal:
                    wrong_on_goal += 1

                # Sampling action based on valid_dist
                a = validActions[torch.multinomial(valid_dist, 1, replacement=True).item()]
                train_val = 1.0
            
            _, r, _, _, on_goal,blocking,_ = gameEnv._step((agentID, a),episode=episode)

            


            s1           = gameEnv._observe(agentID)
            validActions = gameEnv._listNextValidActions(agentID, a,episode=episode)
            d            = gameEnv.finished
            episode_buffer.append([s[0],a,r,s1,d,v[0,0],train_valid,pred_on_goal,int(on_goal),pred_blocking,int(blocking),s[1],train_val])
            rollout=episode_buffer[0]
        
                 
            observations = rollout[0]
            goals=rollout[-2]
            actions = rollout[1]
            rewards = rollout[2]
            values = rollout[5]
            valids = rollout[6]
            blockings = rollout[10]
            on_goals=rollout[8]
            train_value = rollout[-1]
            episode_reward += r

            if r>0:
                    RewardNb += 1 #if it is a positive reward means finished increase RewardNb by 1
            if d == True: #finished
                    print('\n{} Goodbye World. We did it!'.format(step), end='\n')
                    break
        
            
            
            









            
           
            

            model.train()
           
            
            # Compute advantage

            # actions = torch.tensor(actions,dtype=torch.int32) #you don’t need to define placeholders explicitly; you can create tensors directly
            actions_onehot = torch.nn.functional.one_hot(torch.tensor(actions), num_classes=a_size)
            # advantages = torch.tensor(rewards,dtype=torch.float32)
            advantages = rewards + 0.95 * values
            responsible_outputs = torch.sum(a_dist * actions_onehot, dim=1)
            policy_loss = -torch.sum(torch.log(torch.clamp(responsible_outputs, min=1e-15, max=1.0)) * advantages)
            
            # loss=model.evaluate_actions(inputs=inputs,goal_pos=goal_pos,actions=actions,rewards=rewards,state_init=state_out,training=True)[0]

            # print("loss=",loss)
            
            # Compute loss
            optimizer.zero_grad()


            policy_loss.backward()
            
           
            
            optimizer.step()



            #save model

            torch.save(model.state_dict(), model_path)

            
        
            episode_reward += r
            
        
        print(f"Episode {episode}, Total Reward: {episode_reward}")
        print(f"Episode {episode}, Total Policy Loss: {policy_loss}")
        
