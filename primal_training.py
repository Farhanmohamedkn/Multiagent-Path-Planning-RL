# import tensorflow as tf
# from ACNet import ACNet
import torch
import torch.optim as optim
import wandb
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing import Value
import numpy as np

import matplotlib.pyplot as plt
import json
import os
import mapf_gym_cap as mapf_gym
import time
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
from ACNet_pytorch_my_change import ACNet

RNN_SIZE= 512




class PRIMAL(object):
    '''
    This class provides functionality for running multiple instances of the 
    trained network in a single environment
    '''
    def __init__(self,model_path,grid_size):
        self.grid_size=grid_size
        
        self.network=ACNet("global",5,None,False,grid_size,"global")

    
        
    def set_env(self,gym):
        self.num_agents=gym.num_agents
        self.agent_states=[]
        for i in range(self.num_agents):
            rnn_state = (torch.zeros(RNN_SIZE).unsqueeze(0),torch.zeros(RNN_SIZE).unsqueeze(0))
            self.agent_states.append(rnn_state)
        self.size=gym.SIZE
        self.env=gym
        
    def step_all_parallel(self):
        action_probs=[None for i in range(self.num_agents)]
        '''advances the state of the environment by a single step across all agents'''
        #parallel inference
        actions=[]
        inputs=[]
        goal_pos=[]
        for agent in range(1,self.num_agents+1):
            o=self.env._observe(agent)
            inputs.append(o[0])
            goal_pos.append(o[1])
        inputs=torch.tensor(inputs,dtype=torch.float32)
        goal_pos=torch.tensor(goal_pos,dtype=torch.float32)
        #compute up to LSTM in parallel
        h3_vec=self.network.h3=self.network(inputs=inputs,goal_pos=goal_pos,state_in=self.agent_states,training=True)
        
        h3_vec=h3_vec[0]
        rnn_out=[]
        #now go all the way past the lstm sequentially feeding the rnn_state
        for a in range(0,self.num_agents):
            rnn_state=self.agent_states[a]
            lstm_output,state = self.network(inputs=inputs[a],goal_pos=goal_pos[a],state_in=self.agent_states[a],training=True)
            rnn_out.append(lstm_output[0])
            self.agent_states[a]=state
        #now finish in parallel
        policy_vec=self.policy
            
        # policy_vec=self.sess.run([self.network.policy], 
                                        #  feed_dict={self.network.rnn_out:rnn_out})
        policy_vec=policy_vec[0]
        for agent in range(1,self.num_agents+1):
            action=torch.argmax(policy_vec[agent-1])
            self.env._step((agent,action))
          
    def find_path(self,max_step=256):
        '''run a full environment to completion, or until max_step steps'''
        solution=[]
        step=0
        while((not self.env._complete()) and step<max_step):
            timestep=[]
            for agent in range(1,self.env.num_agents+1):
                timestep.append(self.env.world.getPos(agent))
            solution.append(np.array(timestep))
            self.step_all_parallel()
            step+=1
            #print(step)
        if step==max_step:
            raise OutOfTimeError
        for agent in range(1,self.env.num_agents):
            timestep.append(self.env.world.getPos(agent))
        return np.array(solution)
    

    
def run_training(gameEnv,num_agents,network,optimizer):
    max_step     = 256
    episode_count          = 0
    while (episode_count<1000):
        agent_states=[]
        for i in range(num_agents):
            rnn_state = (torch.zeros(RNN_SIZE).unsqueeze(0),torch.zeros(RNN_SIZE).unsqueeze(0))
            agent_states.append(rnn_state)
            step=0
            while((not gameEnv._complete()) and step<max_step):
                
                for agent in range(1,num_agents+1):
                    actions=[]
                    inputs=[]
                    goal_pos=[]
                    for agent in range(1,gameEnv.num_agents+1):
                        o=gameEnv._observe(agent)
                        inputs.append(o[0])
                        goal_pos.append(o[1])
                    inputs=torch.tensor(inputs,dtype=torch.float32)
                    goal_pos=torch.tensor(goal_pos,dtype=torch.float32)
                    
                    step_all_parallel()
                step+=1
            

    
    

   
   
       
 # Learning parameters
gamma                  = .95 # discount rate for advantage estimation and reward discounting

EXPERIENCE_BUFFER_SIZE = 20
GRID_SIZE              = 10 #the size of the FOV grid to apply to each agent
ENVIRONMENT_SIZE       = (10,20)#the total size of the environment (length of one side)
OBSTACLE_DENSITY       = (0,.2) #range of densities
DIAG_MVMT              = False # Diagonal movements allowed?
a_size                 = 5 + int(DIAG_MVMT)*4
SUMMARY_WINDOW         = 10


EPISODE_SAMPLES        = EXPERIENCE_BUFFER_SIZE # 64
LR_Q                   = 2e-5  #8.e-5 / NUM_THREADS # default: 1e-5
ADAPT_LR               = True
ADAPT_COEFF            = 5.e-5 #the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
load_model             = False
RESET_TRAINER          = False
model_path             = "/home/noushad/Master_thesis/Primal_Thesis/pytorch_model"
gifs_path              = 'gifs_primal'
train_log_path         = '/home/noushad/Master_thesis/Primal_Thesis/train_log_path'

# writer = SummaryWriter(log_dir=train_log_path)

GLOBAL_NET_SCOPE       = 'global'

#Imitation options
PRIMING_LENGTH         = 0    # number of episodes at the beginning to train only on demonstrations
DEMONSTRATION_PROB     = 0.2  # probability of training on a demonstration per episode

# Simulation options
FULL_HELP              = False
OUTPUT_GIFS            = False
SAVE_EPISODE_BUFFER    = False   

if __name__ == "__main__":

    

    num_agents = 4
    gym=mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE, 
                                   observation_size=GRID_SIZE,PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)
    network=ACNet(GLOBAL_NET_SCOPE,5,None,False,GRID_SIZE,GLOBAL_NET_SCOPE)
    optimizer=torch.optim.NAdam(network.parameters(), lr=LR_Q)

    

    print("Starting training for %d agents" % num_agents)
        
            
    run_training(gym,num_agents,network,optimizer)

print("finished training")
