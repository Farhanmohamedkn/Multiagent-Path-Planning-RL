from __future__ import division

import gym
import numpy as np
import random
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
import wandb
import torch.nn as nn
import torch.multiprocessing as mp
from shared_optim import SharedAdam

class SharedGameEnv:
    
                                
    def __init__(self, num_agents, DIAGONAL_MOVEMENT, SIZE,observation_size,PROB, FULL_HELP):
        self.env = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAGONAL_MOVEMENT, SIZE=SIZE, 
                                observation_size=observation_size,PROB=PROB, FULL_HELP=FULL_HELP)
        self.num_agents = num_agents+1
        
        # self.init_observation   = [self.env._observe(agent_id) for agent_id in range(1,self.num_agents)]
        
        

        # self.init_validActions  = [self.env._listNextValidActions(agent_id) for agent_id in range(1,self.num_agents)]

    def init_observations_for_multiprocessing(self):
        self.init_observation   = [self.env._observe(agent_id) for agent_id in range(1,self.num_agents)]
        
        cur_pos=self.env.getPositions()
        goals=self.env.getGoals()
        print("Current pos of all agents",agent_id,cur_pos,goals)
        

        self.init_validActions  = [self.env._listNextValidActions(agent_id) for agent_id in range(1,self.num_agents)]
        # Convert each observation to a tensor and share memory
        tensor_inputs   = []
        tensor_goal_pos = []
        for obs in self.init_observation: # we have to share data explicitly for multiprocess
            numpy_array= np.array(obs[0]) #to make the numpy to tensor conversion faster
            inputs = torch.tensor(numpy_array)
            goal_pos=torch.tensor(obs[1])
            inputs.share_memory_()
            goal_pos.share_memory_()
            
            tensor_inputs.append(inputs)
            tensor_goal_pos.append(goal_pos)
        return tensor_inputs,tensor_goal_pos
    
    def init_valid_action_for_multiprocessing(self):
          # Convert each valid action for each agent to a tensor and share memory
        tensor_actions   = []
        for actions in self.init_validActions:
            inputs = torch.tensor(actions)
            
            inputs.share_memory_()
            
            
            tensor_actions.append(inputs)
           
        return tensor_actions
    def get_observation_after_step_taken(self,agent_id):
        observation   = self.env._observe(agent_id)
        
        return observation
    
    def step_all_for_multiprocessing(self,agent_id,action,episode_count):

        new_obs, reward, done, nextActions, on_goal, blocking, valid_action = self.env._step((agent_id, action),episode=episode_count)

        return new_obs,reward, done, nextActions, on_goal, blocking, valid_action
    
    def get_next_valid_actions(self,agent_id,previous_action,episode_count):
         
         next_valid_actions=self.env._listNextValidActions(agent_id, previous_action,episode=episode_count)

         return next_valid_actions
    
    def reset_environment(self,agent_id):
         
         self.env._reset(agent_id)

    def get_cur_pos_of_agent(self,agent_id):
         
         cur_pos=self.env.world.getPos(agent_id)

         return cur_pos
    
    def is_all_agent_reached_goal(self):
         
         cur_pos=self.env.getPositions()
         goals=self.env.getGoals()
        #  print("Cur pos",cur_pos)
        #  print("goals",goals)

         finished=self.env.finished

         return finished
    
def worker(agent_id,shared_env,episode_count,lock,barrier):
    
        episode_count=0
        while episode_count<1:
            print("New Episode started=",episode_count)
            # if agent_id==1:
                    # print("Environment reseted")
                    # shared_env.reset_environment(agent_id)
            
            barrier.wait() #wait for all agents to reset


            inputs_list,goal_pos_list=shared_env.init_observations_for_multiprocessing()
            inputs=inputs_list[agent_id-1]
            goal_pos=goal_pos_list[agent_id-1]
            # print("agent=",agent_id,"observation initialized",goal_pos)
            # print("length of observation before=",len(inputs_list))

            action_list_for_all_agent=shared_env.init_valid_action_for_multiprocessing()
            validActions=action_list_for_all_agent[agent_id-1]
            # print("agent=",agent_id,"action available initialized=",validActions)
            # Generate a random index
            random_index = torch.randint(0, (len(validActions)-1), (1,)).item()
            random_int=validActions[random_index].item()
            # print("random index,",random_int)
            optimal_action=int(validActions[random_int])
            episode_step_count=0
            reached_goal=False
            with lock:
                p=shared_env.get_cur_pos_of_agent(agent_id)
                print("initial pos of agent",agent_id,"is",p)
            barrier.wait() #make sure all agents initialized properly
            
            
            while (episode_step_count<3):
                
                with lock:
                #   print("agent=",agent_id,"action taken=",optimal_action)
                  new_obs,reward, done, nextActions, on_goal, blocking, valid_action=shared_env.step_all_for_multiprocessing(agent_id,optimal_action,episode_step_count)
                  #it returns the same observation AND validaction as before step
                
                barrier.wait() #wait till all agents take a step
                with lock:
                        observation= shared_env.get_observation_after_step_taken(agent_id) #new_observation after all agent take steps
                        input=np.array(observation[0])
                        input=torch.tensor(input, dtype=torch.float32)
                        
                        cur_pos=shared_env.get_cur_pos_of_agent(agent_id)
                        print("cur pos of agent",agent_id,"is",cur_pos)
                        shared_env.init_observations_for_multiprocessing()

                        

                        goal_pos=observation[1]
                        goal_pos=torch.tensor(goal_pos, dtype=torch.float32)
                        # print("agent=",agent_id,"observation after step taken",goal_pos)

                        # optimal_action=shared_env.get_action_after_step_taken(agent_id)

                        # Select a random value from the list
                        next_valid_action=shared_env.get_next_valid_actions(agent_id,optimal_action,episode_step_count)
                        random_value = random.randint(0, len(next_valid_action) - 1)
                        optimal_action=next_valid_action[random_value]
                        reached_goal=shared_env.is_all_agent_reached_goal() #check all agents reached goal or not
                        if reached_goal==True:
                             print("Good bye world we did it")
                             break
                        # print("agent=",agent_id,"action available after step=",next_valid_action)

                # with lock:
                    
                    # episode_count.value += 1.0/num_agents
                    # print("after",episode_count)
                barrier.wait() #wait for all agent to inrease the steps taken 
                episode_step_count+=1
                print("agent",agent_id,"increased episode counts=",episode_step_count)
                barrier.wait() #wait for all agent to inrease the steps taken
            episode_count+=1





    


if __name__ == '__main__':
    num_agents = 2
    GRID_SIZE              = 10 #the size of the FOV grid to apply to each agent
    ENVIRONMENT_SIZE       = (10,10)#the total size of the environment (length of one side)
    OBSTACLE_DENSITY       = (0,.2) #range of densities
    DIAG_MVMT              = False # Diagonal movements allowed?
    a_size                 = 5 + int(DIAG_MVMT)*4
    FULL_HELP              = False
    sharedenv = SharedGameEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE, 
                                observation_size=GRID_SIZE,PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)
    
    # tensor_observation = sharedenv.prepare_observations_for_multiprocessing()

    episode_count=mp.Value('i', 0)
    lock = mp.Lock()
    barrier = mp.Barrier(num_agents)
    
    
    processes = []
    for agent_id in range(1,num_agents+1):
        print(f'worker {agent_id}: started')
        p = mp.Process(target=worker, args=(agent_id, sharedenv,episode_count,lock,barrier))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
