# import tensorflow as tf
# from ACNet import ACNet
import torch
import torch.optim as optim
import wandb
import torch.nn as nn
import torch.nn.functional as F
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

results_path="primal_results"
environment_path="saved_environments"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load numpy array from .npy file
# data = np.load('/home/noushad/Master Thesis/PRIMAL/saved_environments/4_agents_10_size_0_density_id_0_environment.npy')


# Print shape and data type of the array
# print("Shape of the array:", data.shape)
# print("Data type of the array:", data.dtype)
model_path             = "/home/noushad/Master_thesis/Primal_Thesis/pytorch_model/model-4000.0.pt"
RNN_SIZE= 512




class PRIMAL(object):
    '''
    This class provides functionality for running multiple instances of the 
    trained network in a single environment
    '''
    def __init__(self,model_path,grid_size):
        self.grid_size=grid_size
        # config = tf.compat.v1.ConfigProto(allow_soft_placement = True)
        # config.gpu_options.allow_growth=True
        # self.sess=tf.compat.v1.Session(config=config)
        self.network=ACNet("global",5,None,False,grid_size,"global")

        pytorch_model = torch.load(model_path)
        # print(pytorch_model.keys())

        # Load the model state_dict (weights)
        self.network.load_state_dict(pytorch_model['model_state_dict'])
        self.network.eval()

    
        
    def set_env(self,gym):
        self.num_agents=gym.num_agents
        self.agent_states=[]
        # for i in range(self.num_agents):
            # rnn_state = (torch.zeros(1,RNN_SIZE),torch.zeros(1,RNN_SIZE))
            # self.agent_states.append(rnn_state)

            
        # h_0=torch.stack([state[0] for state in self.agent_states])
        # c_0=torch.stack([state[1] for state in self.agent_states])
        # self.h_0=h_0
        # self.c_0=c_0
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
        inputs=torch.stack([torch.tensor(observation,dtype=torch.float32) for observation in inputs])
        goal_pos=torch.stack([torch.tensor(goal,dtype=torch.float32) for goal in goal_pos])
        #compute up to LSTM in parallel
        for i in range(self.num_agents):
            rnn_state = (torch.zeros(1,RNN_SIZE),torch.zeros(1,RNN_SIZE))

            _,_,_,_,h3_vec,_,_,_=self.network(inputs=inputs,goal_pos=goal_pos,state_in=rnn_state,training=False)
            self.agent_states.append(rnn_state)
        # h3_vec = self.sess.run([self.network.h3], 
                                        #  feed_dict={self.network.inputs:inputs,
                                                    # self.network.goal_pos:goal_pos})
        h3_vec=h3_vec[0]
        h3_vec=h3_vec.unsqueeze(0)
        rnn_out=[]
        #now go all the way past the lstm sequentially feeding the rnn_state
        for a in range(0,self.num_agents):
            rnn_state=self.agent_states[a]
            lstm_output,state = self.network.lstm(h3_vec,self.agent_states[a])
            # lstm_output,state = self.sess.run([self.network.rnn_out,self.network.state_out], 
                                        #  feed_dict={self.network.inputs:[inputs[a]],
                                                    # self.network.h3:[h3_vec[a]],
                                                    # self.network.state_in[0]:rnn_state[0],
                                                    # self.network.state_in[1]:rnn_state[1]})
            rnn_out.append(lstm_output[0])
            self.agent_states[a]=state
        #now finish in parallel
        rnn_out= torch.stack(rnn_out)   
        policy_vec=F.softmax(self.network.policy_layer(rnn_out))
            
        # policy_vec=self.sess.run([self.network.policy], 
                                        #  feed_dict={self.network.rnn_out:rnn_out})
        
        # Applying torch.argmax
        # max_index = torch.argmax(policy_vec)
        # Printing results
        # print("Tensor values:", policy_vec)
        # print("Max index:", max_index)  # Output: tensor(4)
        # print("Max value:", policy_vec[max_index])  # Output: tensor(0.9936)
    
        for agent in range(1,self.num_agents+1):

            # Print the policy vector for the current agent
            # print(f"Agent {agent}: policy_vec[{agent-1}] = {policy_vec[agent-1]}")
            action=torch.argmax(policy_vec[agent-1])
            action=action.item()
            print("agent moved",agent,"action=",action)
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
    
def make_name(n,s,d,id,extension,dirname,extra=""):
    if extra=="":
        return dirname+'/'+"{}_agents_{}_size_{}_density_id_{}{}".format(n,s,d,id,extension)
    else:
        return dirname+'/'+"{}_agents_{}_size_{}_density_id_{}_{}{}".format(n,s,d,id,extra,extension)
    
def run_simulations(next,primal):
    #txt file: planning time, crash, nsteps, finished
    (n,s,d,id) = next
    environment_data_filename=make_name(n,s,d,id,".npy",environment_path,extra="environment")
    world=np.load(environment_data_filename)
    gym=mapf_gym.MAPFEnv(num_agents=n, world0=world[0],goals0=world[1])
    primal.set_env(gym)
    solution_filename=make_name(n,s,d,id,".npy",results_path,extra="solution")
    txt_filename=make_name(n,s,d,id,".txt",results_path)
    world=gym.getObstacleMap()
    start_positions=tuple(gym.getPositions())
    goals=tuple(gym.getGoals())
    start_time=time.time()
    results=dict()
    start_time=time.time()
    try:
        #print('Starting test ({},{},{},{})'.format(n,s,d,id))
        path=primal.find_path(256 + 128*int(s>=80) + 128*int(s>=160))
        results['finished']=True
        results['time']=time.time()-start_time
        results['length']=len(path)
        np.save(solution_filename,path)
    except OutOfTimeError:
        results['time']=time.time()-start_time
        results['finished']=False
    results['crashed']=False
    f=open(txt_filename,'w')
    f.write(json.dumps(results))
    f.close()

if __name__ == "__main__":
#    import sys
#    num_agents = int(sys.argv[1])

    primal=PRIMAL(model_path,10)
    num_agents = 2

    while num_agents < 1024:
        num_agents *= 2

        print("Starting tests for %d agents" % num_agents)
        for size in [10,20,40,80,160]:
            if size==10 and num_agents>32:continue
            if size==20 and num_agents>128:continue
            if size==40 and num_agents>512:continue
            for density in [0,.1,.2,.3]:
                for iter in range(100):
                    run_simulations((num_agents,size,density,iter),primal)
print("finished all tests!")
