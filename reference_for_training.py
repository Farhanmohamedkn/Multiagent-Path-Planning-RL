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

from ACNet_direct_pytorch import ACNet

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



# Define constants
GAMMA = 0.99  # Discount factor
LR = 0.001    # Learning rate
BATCH_SIZE = 32
NUM_EPISODES = 10
MAX_STEPS = 200
#helping function

gifs_path              = 'gifs_primal'

def make_gif(images, fname=gifs_path, duration=2, true_image=False,salience=False,salIMGS=None):
    if len(images) == 0:
        raise ValueError("The images list is empty. Provide a list of images to create a GIF.")
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")




# Define the SimpleActorCritic model
    

class SimpleActorCritic(nn.Module):
    def __init__(self, observation_shape, action_size):
        super(SimpleActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Adjusted input size
        
        self.actor_fc = nn.Linear(256, action_size)
        self.critic_fc = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        action_probs = F.softmax(self.actor_fc(x), dim=-1)
        state_value = self.critic_fc(x)
        
        return action_probs, state_value

# class SimpleActorCritic(nn.Module):
#     def __init__(self, observation_size, action_size):
#         super(SimpleActorCritic, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
#         self.fc1 = nn.Linear(64 * (observation_size - 4) * (observation_size - 4), 256)
        
#         self.actor_fc = nn.Linear(256, action_size)
#         self.critic_fc = nn.Linear(256, 1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
        
#         action_probs = F.softmax(self.actor_fc(x), dim=-1)
#         state_value = self.critic_fc(x)
        
#         return action_probs, state_value
    
#We create an instance of NeuralNetwork, and move it to the device, and print its structure.
# model = SimpleActorCritic().to(device)
# print(model)    




# Initialize environment
num_agents = 5  # Set the number of agents
DIAG_MVMT = False  # Allow diagonal movements
ENVIRONMENT_SIZE = (10,20)  # Size of the grid environment
GRID_SIZE = 10
OBSTACLE_DENSITY = (0,.01)
FULL_HELP = False
LR=1e-5
RNN_SIZE = 512

agent_id=1




# print(gameEnv)

# Set up the model, optimizer, and other parameters
a_size = 5 + int(DIAG_MVMT)*4 # Define action size
# model = ACNet('global', a_size, None, True, GRID_SIZE, 'global')
# optimizer = torch.optim.NAdam(model.parameters(), lr=LR)

## Initialize Gym environment


gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE,
                           observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)

# number of agents

num_agents = gameEnv.num_agents

#observation space
state_shape,deltas = gameEnv._observe(agent_id) 
#Deltas are used for unit vector to pint to the goal if it  is not in FOV
poss_map, goal_map, goals_map, obs_map = state_shape
dx,dy,mag=deltas
# print("Agents position in current agent FOV:", poss_map) #position of all agents not a tuple just 1 10x10
# print("curent agent goal in current agent FOV:",goal_map) # sometime empty goal direction of current agent unit vector are provided
# print("Agents goals in current agent FOV:",goals_map) # goals of all agent but not same
# print("obstacle in  of current agent FOV:",obs_map) # goals of all agent but not same

# Processing maps to pytorch tensors from observation
poss_map_tensor = torch.tensor(poss_map,dtype=torch.float32)  # Convert to PyTorch tensor if needed is 1 tensor is fine for 4 values??
goal_map_tensor = torch.tensor(goal_map,dtype=torch.float32)
goals_map_tensor = torch.tensor(goals_map,dtype=torch.float32)
obs_map_tensor = torch.tensor(obs_map,dtype=torch.float32)


goals_map_tensor = torch.tensor(goals_map,dtype=torch.float32)
observation_tensor=torch.stack([poss_map_tensor,goal_map_tensor,goals_map_tensor,obs_map_tensor,])

# # convert the dx,dy,mag to pytorch tensors from observation

# dx_tensor=torch.tensor(dx,dtype=torch.float32)
# dy_tensor=torch.tensor(dy,dtype=torch.float32)
# mag_tensor=torch.tensor(dy,dtype=torch.float32)
# # print("poss_map_tensor:",poss_map_tensor)

def get_next_valid_actions(agent_id,step):
     gameEnv._listNextValidActions(agent_id,episode=step)


def sample_valid_action(action_probs, valid_actions):
    mask = torch.zeros_like(action_probs)
    mask[valid_actions] = 1
    filtered_action_probs = action_probs * mask
    if filtered_action_probs.sum().item() == 0:
        raise ValueError("No valid actions available")
    filtered_action_probs = filtered_action_probs / filtered_action_probs.sum()
    action = torch.multinomial(filtered_action_probs, 1).item()
    return action

#action space

action_size=5 #action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST


# initialize model,optimizer,and loss function
model= SimpleActorCritic(observation_tensor,action_size)
optimizer = torch.optim.NAdam(model.parameters(), lr=LR)
mse_loss=nn.MSELoss()

# print("error:", error)

# Training 

for episode in range(NUM_EPISODES):
        # state = gameEnv._reset(agent_id)
        # state = gameEnv._observe(agent_id)
        
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            action_probs, state_value = model(state)
            validActions   = get_next_valid_actions(agent_id,step)
            action = sample_valid_action(action_probs, validActions)
           
            
            state, reward, done, nextActions, on_goal, blocking, valid_action = gameEnv._step((agent_id, action),episode=step) #action executed by the each agent

            next_state=state
            # next_state = preprocess_observation(next_state)
            
            _, next_state_value = model(next_state)
            
            # Compute advantage
            advantage = reward + (1 - done) * GAMMA * next_state_value - state_value
            
            # Compute loss
            actor_loss = -torch.log(action_probs[action]) * advantage.detach()
            critic_loss = mse_loss(state_value, reward + (1 - done) * GAMMA * next_state_value.detach())
            loss = actor_loss + critic_loss

            print("loss=",loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode}, Total Reward: {episode_reward}")
