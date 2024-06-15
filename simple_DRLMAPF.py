
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

# Define constants
GAMMA = 0.99  # Discount factor
LR = 0.001    # Learning rate
BATCH_SIZE = 32
NUM_EPISODES = 1000
MAX_STEPS = 200
#helping function

gifs_path              = 'gifs_primal'

def make_gif(images, fname=gifs_path, duration=2, true_image=False,salience=False,salIMGS=None):
    if len(images) == 0:
        raise ValueError("The images list is empty. Provide a list of images to create a GIF.")
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")



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
model = ACNet('global', a_size, None, True, GRID_SIZE, 'global')
# optimizer = torch.optim.NAdam(model.parameters(), lr=LR)

## Initialize Gym environment


gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE,
                           observation_size=GRID_SIZE, PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)

# number of agents

num_agents = gameEnv.num_agents

#observation space
# state_shape,deltas = gameEnv._observe(agent_id) 

# s= gameEnv._observe(agent_id) 
# inputs=s[0]
# goal_pos=s[1]
# # Convert to PyTorch tensor of size 2x1
# goal_pos=torch.tensor(goal_pos, dtype=torch.float32)

# print("s[goal_pos]",goal_pos)
# print("s[1]",s[1])

#Deltas are used for unit vector to pint to the goal if it  is not in FOV
# poss_map, goal_map, goals_map, obs_map = state_shape
# combined_array=np.stack([poss_map, goal_map, goals_map, obs_map], axis=0)

# Convert to PyTorch tensor
# state_tensor = torch.from_numpy(combined_array).type(torch.float32)
# Now state_tensor has the desired shape (5, 4, 10, 10)
# print(state_tensor.shape)



# dx,dy,mag=deltas
# print("Agents position in current agent FOV:", poss_map) #position of all agents not a tuple just 1 10x10
# print("curent agent goal in current agent FOV:",goal_map) # sometime empty goal direction of current agent unit vector are provided
# print("Agents goals in current agent FOV:",goals_map) # goals of all agent but not same
# print("obstacle in  of current agent FOV:",obs_map) # goals of all agent but not same

# Processing maps to pytorch tensors from observation
# poss_map_tensor = torch.tensor(poss_map,dtype=torch.float32)  # Convert to PyTorch tensor if needed is 1 tensor is fine for 4 values??
# goal_map_tensor = torch.tensor(goal_map,dtype=torch.float32)
# goals_map_tensor = torch.tensor(goals_map,dtype=torch.float32)
# obs_map_tensor = torch.tensor(obs_map,dtype=torch.float32)


# goals_map_tensor = torch.tensor(goals_map,dtype=torch.float32)
# observation_tensor=torch.stack([poss_map_tensor,goal_map_tensor,goals_map_tensor,obs_map_tensor,])

# # convert the dx,dy,mag to pytorch tensors from observation

# dx_tensor=torch.tensor(dx,dtype=torch.float32)
# dy_tensor=torch.tensor(dy,dtype=torch.float32)
# mag_tensor=torch.tensor(dy,dtype=torch.float32)
# # print("poss_map_tensor:",poss_map_tensor)



#action space

action_size=5 #action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST


# initialize model,optimizer,and loss function
# model= SimpleActorCritic(observation_tensor,action_size)
optimizer = torch.optim.NAdam(model.parameters(), lr=LR)
mse_loss=nn.MSELoss()

# print("error:", error)
# Training parameters
max_episodes = 1000
max_steps = 200

# Training loop
for epoch in range(max_episodes):

    for step in range(max_steps):
        # Render the environment
        # gameEnv._render()
        s= gameEnv._observe(agent_id) 
        inputs=s[0] #pso,goalobs maps
         # Convert to PyTorch tensor of size 4x10x10
        inputs=torch.tensor(inputs, dtype=torch.float32)

        goal_pos=s[1] #dx,dy,mag
        # Convert to PyTorch tensor of size 2x1
        goal_pos=torch.tensor(goal_pos, dtype=torch.float32)
        # print("state",s1)

        # print(error)
        # state_tensor = torch.randn(5, 4, 10, 10)  # Permute to match Conv2d input shape
        print()
        # goal_tensor = torch.randn(5, 3)
        # goal_as_tuples = gameEnv.getGoals()
        # goal_tensor=torch.tensor(goal_as_tuples,dtype=torch.float32)

        with torch.no_grad():
            policy, value, (hxs, cxs), blocking, on_goal = model(inputs, goal_pos)
            



        
        for agent_id in range(1,num_agents+1):


            # print("i=",i)

             # Initial state from the environment
            # if i==1:
            #     test = gameEnv._reset(i)
                # print("Environment reseted after maximum steps taken by each agent")
            episode_frames = [ gameEnv._render(mode='rgb_array',screen_height=900,screen_width=900) ]
            # print("we are here 1st")
            validActions   = gameEnv._listNextValidActions(agent_id,episode=step)
            p              = gameEnv.world.getPos(agent_id)
            on_goal        = gameEnv.world.goals[p[0],p[1]]==agent_id
            best_action = np.argmax(validActions)
            # print("action",best_action)
            a = best_action

            state, reward, done, nextActions, on_goal, blocking, valid_action = gameEnv._step((agent_id, a),episode=step) #action executed by the each agent
            print(f"position of agent {agent_id} at time step",step,gameEnv.world.getPos(agent_id))

            # Get common observation for all agents after all individual actions have been performed
            s1           = gameEnv._observe(agent_id)
            validActions = gameEnv._listNextValidActions(agent_id, a,episode=step)
            d            = gameEnv.finished | done
            
            # episode_frames.append(gameEnv._render(mode='rgb_array',screen_width=900,screen_height=900))
            # print("we are here 2nd",step)
            # images = np.array(episode_frames)
            # print("finished?",d)
            if on_goal==True:
                  print("in epoch number=",epoch,"agent",agent_id,"found goal at step",step)
                  time_per_step = 0.1
                #   make_gif(images,gifs_path,duration=len(images)*time_per_step,true_image=True,salience=False)
                  
            if d == True:
                        print('\n{} Goodbye World. We did it!'.format(step), end='\n')
                        break

            # print("any agent reached  goal?",done)

    #***----Lets print the reward and action the current agent took------**
            
            # if reward == FINISH_REWARD:
            #     print("no idea how it gave",reward)
            # if reward == 1:
            #     print("action executed and reached goal (or stayed on)",reward)
            # if reward == GOAL_REWARD:
            #     print("action executed or stayed still or reached goal ",reward)
            
            # if reward == COLLISION_REWARD:
            #     print("collision with wall,or robot or outof bound ",reward)
            
            # if reward == IDLE_COST:
            #     print("robot stayed",reward)
            # if reward == BLOCKING_COST:
            #     print("robot blocked another robot",reward)
            # if reward == ACTION_COST:
            #     print("robot moved ",reward)
            
            
            # else:
            #     print("reward error",reward)

            
            
            # print("action taken is",a,"recieved reward=",reward,"at time step=",step,"for agent id=",current_agent)
        
            # Convert to tensor
            reward_tensor = torch.tensor(reward, requires_grad=False)
            
            done_tensor = torch.tensor(done, requires_grad=False)
            # print('reward_tensor',reward_tensor, 'done_tensor', done_tensor)
            # Compute loss
            # value_loss = F.mse_loss(value.squeeze(), reward_tensor)
            # print('value_loss',value_loss)
            # advantage = reward_tensor - value.detach().squeeze()
            # policy_loss = -(torch.log(policy.gather(1, torch.tensor(a, requires_grad=False).view(-1, 1))) * advantage.detach()).mean()
            # blocking_loss = F.binary_cross_entropy(blocking.squeeze(), torch.tensor(done))
            # on_goal_loss = F.binary_cross_entropy(on_goal.squeeze(), torch.tensor(done))
            
            # loss = value_loss + policy_loss + blocking_loss + on_goal_loss
            # loss = value_loss + policy_loss 
            # print('loss', loss)
            # Perform backpropagation
            optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            
            # state = next_state
            
            # if Finished:
            #     break
        

    

    state = gameEnv._reset(1) #if agent id is 1 initiate the env
    print("Resets environment")
    # print(f"Episode {episode} finished with total reward {np.sum(reward)}")

print("Training complete")



# %%
