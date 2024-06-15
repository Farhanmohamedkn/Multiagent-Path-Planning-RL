from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn


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

# Define constants
GAMMA = 0.99  # Discount factor
LR = 0.001    # Learning rate
BATCH_SIZE = 32
NUM_EPISODES = 1000
MAX_STEPS = 200

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


class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers for local observation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Fully connected layer for goal direction/distance
        self.fc_goal = nn.Linear(3, 64)  # Assuming goal unit vector has 2 dimensions

        # Concatenation layer
        self.concat_layer = nn.Linear(64 + 64, 128)  # 64 from conv, 64 from fc_goal

        # Additional fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )

        # LSTM cell
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)

        # Output layers
        # self.policy_neurons = nn.Linear(512, num_classes)  # Adjust num_classes as needed
        self.policy_neurons = nn.Linear(512, 5)  # Adjust num_classes as needed
        self.value_output = nn.Linear(512, 1)
        # self.feature_layer = nn.Linear(512, feature_dim)  # Adjust feature_dim as needed
        self.feature_layer = nn.Linear(512, 1)  # Adjust feature_dim as needed

    def forward(self, local_observation, goal_direction):
        # Process local observation through conv layers
        local_features = self.conv_layers(local_observation)
        print("local_features",local_features,local_features.shape)

        # Process goal direction through fc layer
        goal_features = self.fc_goal(goal_direction)
        print("goal_features",goal_features,goal_features.shape)
       

        # Concatenate local features and goal features
        combined_features = torch.cat([local_features.view(local_features.size(0), -1), goal_features], dim=1)

        # Pass through concat layer and additional fc layers
        x = self.concat_layer(combined_features)
        x = self.fc_layers(x)

        # Pass through LSTM
        lstm_output, _ = self.lstm(x.unsqueeze(1))  # Add time dimension

        # Output layers
        policy_output = self.policy_neurons(lstm_output.squeeze(1))
        value_output = self.value_output(lstm_output.squeeze(1))
        feature_output = self.feature_layer(lstm_output.squeeze(1))

        return policy_output, value_output, feature_output

# Instantiate the model
model = CustomNetwork()

gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE,
                           observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)

# Print the model structure
# print(model)
for episode in range(NUM_EPISODES):
    episode_reward = 0
    for step in range(MAX_STEPS):
       
        with torch.no_grad():
             
            state = gameEnv._reset(agent_id)
            s= gameEnv._observe(agent_id) 
            inputs=s[0] #pso,goalobs maps
            # Convert to PyTorch tensor of size 4x10x10
            inputs=torch.tensor(inputs, dtype=torch.float32)
            # inputs=inputs.permute(0, 2, 3, 1)
            goal_pos=s[1] #dx,dy,mag
            # Convert to PyTorch tensor of size 2x1
            goal_pos=torch.tensor(goal_pos, dtype=torch.float32)

            policy_output, value_output, feature_output=model(inputs,goal_pos)
            if step==1:
                previous_action=0
            else:
                previous_action=previous_action

            validActions = gameEnv._listNextValidActions(agent_id, previous_action,episode=step)
            action_probs = policy_output.detach().numpy()
            best_action=np.random.choice(validActions, p=action_probs)
            
            
            state, reward, done, nextActions, on_goal, blocking, valid_action = gameEnv._step((agent_id, best_action),episode=step) #action executed by the each agent

            previous_action=best_action

            next_state=state
            # next_state = preprocess_observation(next_state)
            
            # _, next_state_value = model(next_state)
            
            # # Compute advantage
            # advantage = reward + (1 - done) * GAMMA * next_state_value - state_value
            
            # # Compute loss
            # actor_loss = -torch.log(action_probs[action]) * advantage.detach()
            # critic_loss = mse_loss(state_value, reward + (1 - done) * GAMMA * next_state_value.detach())
            # loss = actor_loss + critic_loss

            # print("loss=",loss)
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        print(f"Episode {episode}, Total Reward: {episode_reward}")
        
        