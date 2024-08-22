
from __future__ import division

import gym
from gym import spaces
import numpy as np
from collections import OrderedDict
from threading import Lock
import sys
from matplotlib.colors import hsv_to_rgb
# import colorsys
import random
import math
import copy
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import NoSolutionError, OutOfTimeError
import torch.multiprocessing as mp
from multiprocessing import Lock, Array
import torch
import gym
import numpy as np
import random
# import tensorflow as tf
# import tensorflow.contrib.layers as layers
# import matplotlib.pyplot as plt
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import threading
import time
import scipy.signal as signal
import os
import GroupLock
import multiprocessing
# import my_env_for_mp as mapf_gym
import pickle
import imageio
from ACNet_pytorch_my_change import ACNet
import torch
import torch.optim as optim
import wandb
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing import Value
from torch.utils.tensorboard import SummaryWriter
from shared_optim import SharedAdam



wandb.login()

wandb.init(project='memory error ')

RNN_SIZE = 512


def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")


# Lock for synchronizing gradient updates
gradient_lock = threading.Lock() #should use it to avaoid race condition to update gradients


def ensure_shared_grads(model, shared_model):
    with gradient_lock:
            for param, shared_param in zip(model.parameters(),shared_model.parameters()):

                if  param.grad is not None:
                    if shared_param.grad is None:
                        # print(param.grad.is_cuda)
                        shared_param.grad = param.grad.clone()
                    # print(shared_param.grad.is_cuda)
                    # print(param.grad.is_cuda)
                    shared_param.grad = param.grad
                    # print("grad param updated")
                # else:
                        # Optionally, you can log or handle the case where param.grad is None
                        # print(f"Warning: Gradients for parameter are None")
            torch.nn.utils.clip_grad_norm_(shared_model.parameters(),max_norm=1.0)


def discount(x, gamma):
    # x_np = x.detach().cpu().numpy()  # Convert tensor to NumPy array
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x,gamma)


#worker agent
# class Worker:
#     def __init__(self,master_network,optimizer, metaAgentID, a_size):
#         # self.workerID = workerID
#         # self.env = game
#         self.metaAgentID = metaAgentID
#         # self.name = "worker_"+str(workerID)
#         # self.agentID = ((workerID-1) % num_workers) + 1 
       

        
#         self.local_AC = ACNet(a_size,True,GRID_SIZE).to(device) #this should be our model?

#         # sync the weights at the beginning
#         self.local_AC.load_state_dict(master_network.state_dict())
        
        

    
        
def train(local_AC,optimizer,master_network,rollout, gamma, bootstrap_value, rnn_state0,episode_count, imitation=False):
    
    
    # if ADAPT_LR:
    #computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
    #we need the +1 so that lr at step 0 is defined
        # lr = torch.div(LR_Q, torch.sqrt(torch.add(1., torch.mul(ADAPT_COEFF, episode_count))))
    # else:
        # lr=LR_Q
    # wandb.log({"current_learning_rate ": lr})
    
    
    if len(rollout)>0: #sometimes the rollout is empty array check why

            if imitation:
                    
                    
                
                    #we calculate the loss differently for imitation
                    #if imitation=True the rollout is assumed to have different dimensions:
                    #[o[0],o[1],optimal_actions]
                    print("imitation mode")
                    inputs_rollout = []
                    goal_pos_rollout = []
                    optimal_actions_rollout = []
                    for item in rollout:
                        
                        inputs=item[0] #dtype=torch.float32
                        
                        goal_pos=item[1] #dtype=torch.float32)
                        
                        
                        optimal_actions=item[2]

                        inputs_rollout.append(inputs)
                        goal_pos_rollout.append(goal_pos)
                        optimal_actions_rollout.append(optimal_actions)
                        
                    
                    
                    global_step=episode_count
                    inputs=np.array(inputs_rollout)
                    inputs=torch.tensor(inputs, dtype=torch.float32).to(device)
                    goal_pos=np.array(goal_pos_rollout)
                    goal_pos=torch.tensor(goal_pos, dtype=torch.float32).to(device)

                

                    local_AC.train()
                    output=local_AC(inputs,goal_pos,state_in=rnn_state0,training=True)
                
                    policy, value, state_out ,state_in, blocking,on_goal,policy_sig,entropy=output
                    optimal_actions  = torch.tensor(optimal_actions_rollout).to(device)
                    optimal_actions_onehot = torch.nn.functional.one_hot(optimal_actions, num_classes=a_size)
                    optimal_actions_onehot =torch.argmax(optimal_actions_onehot, dim=1)
                    optimal_actions_onehot=optimal_actions_onehot.long()

                    loss_function=nn.CrossEntropyLoss()
                    imitation_loss = loss_function(entropy,optimal_actions_onehot)
                    #cant do this directly because of the softmaxissue
                    # self.imitation_loss = torch.sum(self.optimal_actions_onehot*self.policy)

                    
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    imitation_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_AC.parameters(),max_norm=1000)
                    # Manually push the gradients to the global network
                    ensure_shared_grads(local_AC,master_network) 
                    optimizer.step()
                    local_AC.load_state_dict(master_network.state_dict())

                    return imitation_loss
                    
            # print("Shape of rollout:_start", rollout.shape, rollout.size)
            
            observations_rollout = []
            goals_rollout = []
            actions_rollout = []
            rewards_rollout = []
            values_rollout = []
            valids_rollout = []
            blockings_rollout = []
            on_goals_rollout = []
            train_value_rollout = []
            
            for item in rollout:
                observations=item[0]
                goals=item[-2]
                actions=item[1]
                rewards=item[2]
                values = item[5]
                valids = item[6]
                blockings = torch.tensor(item[10])
                on_goals = torch.tensor(item[8])
                train_value=torch.tensor(item[-1])

                observations_rollout.append(observations)
                goals_rollout.append(goals)
                actions_rollout.append(actions)
                rewards_rollout.append(rewards)
                values_rollout.append(values)
                valids_rollout.append(valids)
                blockings_rollout.append(blockings)
                on_goals_rollout.append(on_goals)
                train_value_rollout.append(train_value)
                
            

            # Here we take the rewards and values from the rollout, and use them to 
            # generate the advantage and discounted returns. (With bootstrapping)
            # The advantage function uses "Generalized Advantage Estimation"
            observations1=np.array(observations_rollout,dtype=float)
            goals1=np.array(goals_rollout,dtype=float)
            # observations=torch.tensor(observations,dtype=torch.float32).to(device)
            # goals=torch.tensor(goals,dtype=torch.float32).to(device)

            rewards1=np.array(rewards_rollout,dtype=object)
            if torch.is_tensor(bootstrap_value):
                bootstrap_value = bootstrap_value.cpu().detach().numpy()
                bootstrap_value = bootstrap_value.item()
            
            rewards_plus = np.asarray(rewards1.tolist() + [bootstrap_value])
            if isinstance(rewards_plus, np.ndarray) and any(x is None for x in rewards_plus):
                print("array1 contains None")
            discounted_rewards = discount(rewards_plus,gamma)[:-1]
            # discounted_rewards= torch.stack([torch.tensor(item[-2]) for item in rollout])
            value_temp=[tensor.cpu().detach().numpy() for tensor in values_rollout]
            values1=np.array(value_temp,dtype=object)
            value_plus = np.asarray(values1.tolist() + [bootstrap_value])
            # values_tensor = torch.tensor(values.tolist(), dtype=torch.float32)
            # self.value_plus = values_tensor + bootstrap_value_tensor
            advantages = rewards1 + gamma * value_plus[1:] - value_plus[:-1]
            advantages1 = good_discount(advantages,gamma)


            num_samples = min(EPISODE_SAMPLES,len(advantages1))
            sampleInd = np.sort(np.random.choice(advantages1.shape[0], size=(num_samples,), replace=False))
            advantages2=np.array(advantages1, dtype=np.float32)

            
            # observations = torch.stack(observations).to(device)
            # goals=torch.stack(goals_rollout).to(device)
            local_AC.train()
            inputs1=torch.tensor(observations1,dtype=torch.float32).to(device)
            goal_pos1=torch.tensor(goals1,dtype=torch.float32).to(device)

            # Monitor initial memory usage
            # print(f"Initial allocated memory: {torch.cuda.memory_allocated()} bytes")
            # print(f"Initial reserved memory: {torch.cuda.memory_reserved()} bytes")


            output=local_AC(inputs1,goal_pos1,state_in=rnn_state0,training=True)


            # Monitor memory usage after forward pass
            # print(f"After forward pass allocated memory: {torch.cuda.memory_allocated()} bytes")
            # print(f"After forward pass reserved memory: {torch.cuda.memory_reserved()} bytes")

            policy, value, state_out ,state_in,blocking, on_goal,policy_sig,entropy=output
            


            target_v   = torch.from_numpy(discounted_rewards.copy()).to(device)  #torch.tensor(discounted_rewards)
            actions  = torch.tensor(actions_rollout).to(device)

            actions_onehot = torch.nn.functional.one_hot(actions, num_classes=a_size)
            train_valid = torch.stack(valids_rollout).to(device)
            advantages= torch.from_numpy(advantages2.copy()).to(device) #torch.tensor(advantages)
            train_value=torch.stack(train_value_rollout).to(device)
            target_blockings=torch.stack(blockings_rollout).to(device)
            target_on_goals =torch.stack(on_goals_rollout).to(device)
            responsible_outputs    = torch.sum(policy * actions_onehot,dim=1)
            length_roll=len(observations_rollout) #length of the rollouts
            

            #calculate loss
            value_loss    = torch.sum(train_value * torch.square((target_v-value.view(-1))))
            entropy       = - torch.sum(policy*(torch.log((torch.clamp(policy, min=1e-10, max=1.0)))))
            policy_loss   = torch.sum(advantages*(torch.log((torch.clamp(responsible_outputs, min=1e-15, max=1.0)))))
            valid_loss    = - torch.sum(((torch.log((torch.clamp(policy_sig, min=1e-10, max=1.0))))*train_valid)+((torch.log(torch.clamp(1-policy_sig, min=1e-10, max=1.0)))*(1-train_valid)))
            blocking_loss = - torch.sum(((torch.log((torch.clamp(blocking, min=1e-15, max=1.0))))*target_blockings)+((torch.log((torch.clamp(1-blocking, min=1e-15, max=1.0))))*(1-target_blockings)))
            on_goal_loss  = - torch.sum(((torch.log((torch.clamp(on_goal, min=1e-10, max=1.0))))*target_on_goals)+((torch.log((torch.clamp(1-on_goal, min=1e-10, max=1.0))))*(1-target_on_goals)))
            loss          = (0.5 * value_loss + policy_loss + 0.5*valid_loss - entropy * 0.01 +.5*blocking_loss+0.5*on_goal_loss).to(device)
            
            #calculated loss and apply gradient

            optimizer.zero_grad()
            # print(torch.cuda.max_memory_allocated())
            # print(torch.cuda.memory_allocated())
            # print(torch.cuda.memory_summary())

            loss.backward()
            # Print parameters that have None gradients
            for name, param in local_AC.named_parameters():
                if param.grad is None:
                    print(f"Parameter with None gradient: {name}")

            #  check for gradients
            # for param in self.local_AC.parameters():
                # if param.grad is not None:
                    # print(f"Param: {param.shape}, Grad type: {type(param.grad)}, Grad shape: {param.grad.shape}")



            # Monitor memory usage after backward pass
            # print(f"After backward pass allocated memory: {torch.cuda.memory_allocated()} bytes")
            # print(f"After backward pass reserved memory: {torch.cuda.memory_reserved()} bytes")


            torch.nn.utils.clip_grad_norm_(local_AC.parameters(),max_norm=1000)
            
            # Manually push the gradients to the global network
            
            ensure_shared_grads(local_AC,master_network)  
            
                
            optimizer.step()
                #sync the weights again
            local_AC.load_state_dict(master_network.state_dict())
            g_n=1
            v_n=1
            

            # Update the global network using gradients from loss
            # Generate network statistics to periodically save

            print(agent_id,"trained")
            return value_loss/length_roll, policy_loss /length_roll, valid_loss/length_roll,entropy/length_roll, blocking_loss/length_roll, on_goal_loss/length_roll, g_n, v_n
                

    

def parse_path(path,env):
    '''needed function to take the path generated from M* and create the 
    observations and actions for the agent
    path: the exact path ouput by M*, assuming the correct number of agents
    returns: the list of rollouts for the "episode": 
            list of length num_agents with each sublist a list of tuples 
            (observation[0],observation[1],optimal_action,reward)'''
    result=[[] for i in range(num_workers)]
    for t in range(len(path[:-1])):
        observations=[]
        move_queue=list(range(num_workers))
        for agent in range(1,num_workers+1):
            observations.append(env._observe(agent))
        steps=0
        while len(move_queue)>0:
            steps+=1
            i=move_queue.pop(0)
            o=observations[i]
            pos=path[t][i]
            newPos=path[t+1][i]#guaranteed to be in bounds by loop guard
            direction=(newPos[0]-pos[0],newPos[1]-pos[1])
            a=env.world.getAction(direction)
            state, reward, done, nextActions, on_goal, blocking, valid_action=env._step((i+1,a))
            if steps>num_workers**2:
                #if we have a very confusing situation where lots of agents move
                #in a circle (difficult to parse and also (mostly) impossible to learn)
                return None
            if not valid_action:
                #the tie must be broken here
                move_queue.append(i)
                continue
            result[i].append([o[0],o[1],a])
    return result
        
def work(agent_id,master_network,optimizer,metaAgentID,a_size,max_episode_length,gamma,shared_env,episode_count,lock,barrier):

    #share_memory
    global swarm_reward, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops,episode_wrong_blocking #, episode_invalid_goals

    total_steps, i_buf = 0, 0
    episode_buffers, s1Values = [ [] for _ in range(NUM_BUFFERS) ], [ [] for _ in range(NUM_BUFFERS) ]

    local_AC=ACNet(a_size,True,GRID_SIZE).to(device)
    agentID=agent_id
    env=shared_env

    while episode_count<=20:
        torch.autograd.set_detect_anomaly(True)
            #should implement the cordinator of thread
        
        # sync the weights at the beginning
        local_AC.load_state_dict(master_network.state_dict())
        
        episode_buffer, episode_values = [], []
        episode_reward = episode_step_count = episode_inv_count = 0
        
        d = False

        # Initial state from the environment
        with lock:
            if agentID==1:
                env._reset(agentID)
        barrier.wait()
        # self.synchronize() # synchronize starting time of the threads
        nextvalidActions          = env._listNextValidActions(agentID)
        validActions              = nextvalidActions[agentID]

        # s                     = env._observe(agentID)
        blocking              = False

        cur_pos=env.getPositions()
        cur_goals=env.getGoals()
        print("cur pos and goals agent ",agentID,cur_pos,cur_goals)
        p=getPos(agentID, env.world['agents'])
        # print("helooooooo",p)
        # print(env.world['goals'][p[0],p[1]], agentID)
        on_goal               = env.world['goals'][p[0],p[1]]==agentID
        
        observation           = env._observe(agentID)
        s=observation[agentID]
        # c_in = torch.zeros(1, RNN_SIZE)
        # h_in = torch.zeros(1, RNN_SIZE)
        # rnn_state             =(c_in, h_in)
        inputs=s[0] #observation #pos,goal,obs maps
        inputs=np.array(inputs)
        inputs=torch.tensor(inputs, dtype=torch.float32).to(device)
        inputs=inputs.unsqueeze(0).to(device) #accodomodate batchsize

        goal_pos=s[1] #dx,dy,mag
        goal_pos=np.array(goal_pos)
        goal_pos=torch.tensor(goal_pos, dtype=torch.float32).to(device)
        goal_pos=goal_pos.unsqueeze(0).to(device)
        c_init = torch.zeros(RNN_SIZE).unsqueeze(0).to(device) # should remove from here
        h_init = torch.zeros(RNN_SIZE).unsqueeze(0).to(device)

        state_init = (c_init, h_init)
        
        # _, _, _,_,state_init,_,_,_,_=self.local_AC(inputs=inputs,goal_pos=goal_pos,state_in=state_init,training=True)
        
        rnn_state0            = state_init
        RewardNb = 0 
        wrong_blocking  = 0
        wrong_on_goal=0

        if agentID==1:
            global demon_probs #share_memory
            demon_probs[metaAgentID]=np.random.rand()
        # self.synchronize() # synchronize starting time of the threads

        # reset swarm_reward (for tensorboard)
        swarm_reward[metaAgentID] = 0
        if episode_count.item()<PRIMING_LENGTH and demon_probs[metaAgentID]==10:
            #for the first PRIMING_LENGTH episodes, or with a certain probability
            #don't train on the episode and instead observe a demonstration from M*
            if agentID==1 and episode_count%1000==0:
                # Save the model state dictionary
                torch.save({'model_state_dict': master_network.state_dict()}, f"{model_path}/model-{episode_count}.pt") 
                torch.save({'model_state_dict': local_AC.state_dict()}, f"{model_path}/model-{episode_count+1}.pt")# local or global? should be global right?

                print(f"Model saved at {model_path}/model-{episode_count}.pt")
                # saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                
            global rollouts #shared_memory

            rollouts[metaAgentID]=None
            if(agentID==1):
                world=env.getObstacleMap()
                start_positions=tuple(env.getPositions())
                goals=tuple(env.getGoals())
                try:
                    print("before")
                    mstar_path=cpp_mstar.find_path(world,start_positions,goals,2,5)
                    print("after")
                    rollouts[metaAgentID]=parse_path(mstar_path,env)
                except OutOfTimeError:
                    #M* timed out 
                    print("timeout",episode_count)
                except NoSolutionError:
                    print("nosol????",episode_count,start_positions)
            # self.synchronize()
            if rollouts[metaAgentID] is not None:
                i_l=train(local_AC,optimizer,rollouts[metaAgentID][agentID-1], gamma, None, rnn_state0,episode_count,imitation=True)

                episode_count+=1./num_workers
                if agentID==1:
                    # print("imitation loss wrote to the summary")
                    # Log the imitation loss with the tag 'Losses/Imitation loss'
                    wandb.log({"Losses/Imitation_loss ": i_l})
                    # writer.add_scalar('Losses/Imitation_loss', i_l, episode_count)
                    # writer.flush()
                    # writer.close()
                    
                continue
            continue
        saveGIF = False
        if OUTPUT_GIFS and agentID == 1 and ((not TRAINING) or (episode_count >= nextGIF)):
            saveGIF = True
            nextGIF =episode_count + 64
            GIF_episode = int(episode_count)
            episode_frames = [ env._render(mode='rgb_array',screen_height=900,screen_width=900) ]
            
        while (not env.finished): # Give me something!
            #Take an action using probabilities from policy network output.
            inputs=s[0] #observation #pos,goal,obs maps
            inputs=np.array(inputs)
            inputs=torch.tensor(inputs, dtype=torch.float32).to(device)
            inputs=inputs.unsqueeze(0) #accodomodate batchsize

            goal_pos=s[1] #dx,dy,mag
            goal_pos=np.array(goal_pos)
            goal_pos=torch.tensor(goal_pos, dtype=torch.float32).to(device)
            goal_pos=goal_pos.unsqueeze(0)
            
            with lock:
                local_AC.eval() # Set model to evaluation mode
                with torch.no_grad():
                
                    a_dist, v, rnn_state,state_in,pred_blocking,pred_on_goal,policy_sig,entropy=local_AC(inputs=inputs,goal_pos=goal_pos,state_in=state_init,training=True)
            #there is an other way using no_grad() instead of detach
            
            barrier.wait()


                # Check if the argmax of a_dist is in validActions
            if not (torch.argmax(a_dist.flatten()).item() in validActions):
                episode_inv_count += 1

                # Initialize train_valid with zeros and set valid actions
            train_valid = torch.zeros(a_size)
            train_valid[validActions] = 1
                # Initialize train_valid with zeros and set valid actions
            train_valid = torch.zeros(a_size)
            train_valid[validActions] = 1

            # Get the valid action distribution and normalize it
            # print(a_dist,[0, validActions])
            valid_dist = a_dist[0, validActions]
            valid_dist /= torch.sum(valid_dist)

            if TRAINING:
                if (pred_blocking.flatten()[0] < 0.5) == blocking:
                    wrong_blocking += 1
                if (pred_on_goal.flatten()[0] < 0.5) == on_goal:
                    wrong_on_goal += 1
                a           = validActions[torch.multinomial(valid_dist, 1, replacement=True).item()]
                train_val   = 1.
            else:
                a         = torch.argmax(a_dist.flatten())
                if a not in validActions or not GREEDY:
                    a     = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
                train_val = 1.

            barrier.wait()
            print("before step agent=",agentID,env.getPositions())

            # a=a.item() # to pass it as an integer instead of a tensor
            with lock:
                 _, r, _, _, on_goal,blocking,_,act_status = env._step((agentID, a.item()),episode=episode_count.item())
                 if on_goal:
                     print("agent " ,agentID,"reaaaaaaaaaaaached goaaaalllliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
                 print("agent took" ,agentID,"action if 0 not moved",a.item(),"action executed if 0 or 1 or 2 =", act_status, "episode",episode_count.item(),"all goals=",env.getGoals())

            barrier.wait()
            print("after step agent=",agentID,env.getPositions())
            # print("after step agent current pos=",agentID,getPos(agentID, env.world['agents']))

            

            # self.synchronize() # synchronize threads

            # Get common observation for all agents after all individual actions have been performed
            observation           = env._observe(agentID) #returns mp dictionary
            
            s1           = observation[agentID]
            
        
            nextvalidActions = env._listNextValidActions(agentID, a.item(),episode=episode_count.item())
            validActions              = nextvalidActions[agentID]
        
            d            = env.finished
            barrier.wait()
            # print("aftereeee after agent=",agentID,env.getPositions())

            if saveGIF:
                episode_frames.append(env._render(mode='rgb_array',screen_width=900,screen_height=900))

            # print(on_goal)
            # print(a)
            # hi=int(on_goal)
            
            # train_valid=1 
            episode_buffer.append([s[0],a,r,s1,d,v[0,0],train_valid,pred_on_goal,int(on_goal),pred_blocking,int(blocking),s[1],train_val])
            episode_values.append(v[0,0])
            episode_reward += r
            s = s1
            total_steps += 1
            episode_step_count += 1

            if r>0:
                RewardNb += 1
            if d == True:
                # agents, agents_past, agent_goals = mapf_gym.State.scanForAgents(self) #to see agent actually moved or not
                # print( agents, agents_past, agent_goals)
                print('\n{} Goodbye World. We did it!'.format(episode_step_count), end='\n')

            # If the episode hasn't ended, but the experience buffer is full, then we
            # make an update step using that experience rollout.
            if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                print("yes")
                # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                    episode_buffers[i_buf] = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                else:
                    episode_buffers[i_buf] = episode_buffer[:]

                if d:
                    s1Values[i_buf] = 0
                else:
                    inputs=s[0] #observation #pos,goal,obs maps
                    inputs=np.array(inputs)
                    inputs=torch.tensor(inputs, dtype=torch.float32).to(device) #it should be local_AC.inputs:np.array([s[0]])
                    inputs=inputs.unsqueeze(0)
                    goal_pos=s[1] #dx,dy,mag
                    goal_pos=np.array(goal_pos)
                    goal_pos=torch.tensor(goal_pos, dtype=torch.float32).to(device)
                    goal_pos=goal_pos.unsqueeze(0)
                    _,s1Values[i_buf],_,_,_,_,_,_=local_AC(inputs=inputs,goal_pos=goal_pos,state_in=rnn_state,training=True)
                    s1Values[i_buf]=s1Values[i_buf][0]

                    


                if (episode_count-EPISODE_START) < NUM_BUFFERS:
                    i_rand = np.random.randint(i_buf+1)
                else:
                    
                    
                    i_rand = np.random.randint(NUM_BUFFERS)
            
                    
                v_l,p_l,valid_l,e_l,b_l,og_l,g_n,v_n = train(local_AC,optimizer,master_network,episode_buffers[i_rand],gamma,s1Values[i_rand],rnn_state0,episode_count)

                i_buf = (i_buf + 1) % NUM_BUFFERS
                rnn_state0             = (rnn_state[0].clone().detach(),rnn_state[1].clone().detach())
                episode_buffers[i_buf] = []
            barrier.wait()
            # self.synchronize() # synchronize threads
            
            # sync the weights at the end also???
            print("buffer not full but step finished")
            local_AC.load_state_dict(master_network.state_dict())

            # sess.run(self.pull_global)
            # if episode_step_count >= max_episode_length or d:
            #     break
            if episode_step_count >= 20 or d:
                break

            # if episode_count.value ==2:
                # break
            # with lock:
                # print("step incremented")
                
                
                # episode_step_count+=1./num_agents
               
            print("episode_step_count for agent is",agent_id,episode_step_count)
        barrier.wait()
        print(" complete step took loop break")
        barrier.wait()
        episode_lengths[metaAgentID].append(episode_step_count)
        episode_values=[t.detach() for t in episode_values]

        # episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
        episode_invalid_ops[metaAgentID].append(episode_inv_count)
        episode_wrong_blocking[metaAgentID].append(wrong_blocking)

        # Periodically save gifs of episodes, model parameters, and summary statistics.
        if episode_count.item() % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
            print('                                                                                   ', end='\r')
            print('{} Episode terminated ({},{})'.format(episode_count, agentID, RewardNb), end='\r')

        swarm_reward[metaAgentID] += episode_reward

        # self.synchronize() # synchronize threads

        episode_rewards[metaAgentID].append(swarm_reward[metaAgentID])

        if not TRAINING:
            mutex.acquire()
            if episode_count < NUM_EXPS:
                plan_durations[episode_count] = episode_step_count
            if self.workerID == 1:
                episode_count += 1
                print('({}) Thread {}: {} steps, {:.2f} reward ({} invalids).'.format(episode_count, self.workerID, episode_step_count, episode_reward, episode_inv_count))
            GIF_episode = int(episode_count)
            mutex.release()
        else:
            # print("Episode count=",episode_count)
            with lock:
                
                episode_count+=1/num_agents
                print("episode updated",episode_count)
            # episode_count+=1./num_workers
            if episode_count % 1000 == 0:
                    print("here")
                    print ('Saving Model', end='\n')
                    torch.save({'model_state_dict': master_network.state_dict()}, f"{model_path}/model-{episode_count}.pt") # local or global? should be global right?
                    torch.save({'model_state_dict': local_AC.state_dict()}, f"{model_path}/model-{episode_count+1}.pt")
                    print ('Saved Model', end='\n')
                    
            if episode_count % SUMMARY_WINDOW == 0:
                if episode_count % 1000 == 0:
                    print("here")
                    print ('Saving Model', end='\n')
                    torch.save({'model_state_dict': master_network.state_dict()}, f"{model_path}/model-{episode_count}.pt") # local or global? should be global right?
                    torch.save({'model_state_dict': local_AC.state_dict()}, f"{model_path}/model-{episode_count+1}.pt")
                    print ('Saved Model', end='\n')
                SL = SUMMARY_WINDOW * num_agents
                mean_reward = np.nanmean(episode_rewards[metaAgentID][-SL:])
                mean_length = np.nanmean(episode_lengths[metaAgentID][-SL:])
                # mean_value = np.nanmean(episode_mean_values[self.metaAgentID][-SL:])
                mean_invalid = np.nanmean(episode_invalid_ops[metaAgentID][-SL:])
                mean_wrong_blocking = np.nanmean(episode_wrong_blocking[metaAgentID][-SL:])
                # current_learning_rate = sess.run(lr,feed_dict={global_step:episode_count}) calculate current learning rate

            
                # writer.add_scalar('Perf/Learning Rate', current_learning_rate,episode_count)
                wandb.log({"Perf/Reward": mean_reward, "episode_count": episode_count})
                # wandb.log({"current_learning_rate ": lr})
                wandb.log({"Perf/Length": mean_length})
                wandb.log({"Perf/Valid Rate": ((mean_length-mean_invalid)/mean_length)})
                wandb.log({"Perf/Blocking Prediction Accuracy": ((mean_length-mean_wrong_blocking)/mean_length)})
                wandb.log({"Losses/Value Loss": v_l})
                wandb.log({"Losses/Policy Loss": p_l})
                wandb.log({"Losses/Blocking Loss": b_l})
                wandb.log({"Losses/On Goal Loss": og_l})
                wandb.log({"Losses/Valid Loss": valid_l})

                # writer.add_scalar('Perf/Reward', mean_reward,episode_count)
                # writer.add_scalar('Perf/Length', mean_length,episode_count)

                # writer.add_scalar('Perf/Valid Rate',((mean_length-mean_invalid)/mean_length),episode_count)

                # writer.add_scalar('Perf/Length', ((mean_length-mean_wrong_blocking)/mean_length),episode_count)

                # writer.add_scalar('Losses/Value Loss', v_l,episode_count)
                
                # writer.add_scalar('Losses/Policy Loss', p_l,episode_count)
                # writer.add_scalar('Losses/Blocking Loss', b_l,episode_count)
                # writer.add_scalar('Losses/On Goal Loss', og_l,episode_count)
                # writer.add_scalar('Losses/Valid Loss', valid_l,episode_count)
                # writer.add_scalar('Losses/Grad Norm', g_n,episode_count)
                # writer.add_scalar('Losses/Var Norm', v_n,episode_count)
                # writer.flush()
                

                if printQ:
                    print('{} Tensorboard updated ({})'.format(episode_count, self.workerID), end='\r')

        if saveGIF:
            # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
            time_per_step = 0.1
            images = np.array(episode_frames)
            if TRAINING:
                make_gif(images, '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode,episode_step_count,swarm_reward[self.metaAgentID]))
            else:
                make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path,GIF_episode,episode_step_count), duration=len(images)*time_per_step,true_image=True,salience=False)
        if SAVE_EPISODE_BUFFER:
            with open('gifs3D/episode_{}.dat'.format(GIF_episode), 'wb') as file:
                pickle.dump(episode_buffer, file)


###-- Training --###




# if not TRAINING:
#     plan_durations = np.array([0 for _ in range(NUM_EXPS)])
#     mutex = threading.Lock()
#     gifs_path += '_tests'
    # if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        # os.makedirs('gifs3D')

#Create a directory to save episode playback gifs to
# if not os.path.exists(gifs_path):
    # os.makedirs(gifs_path)


    
if __name__=='__main__':

    

    # mp.set_start_method('spawn', force=True)

    # Learning parameters
    max_episode_length     = 256
    episode_count          = 0
    EPISODE_START          = episode_count
    gamma                  = .95 # discount rate for advantage estimation and reward discounting

    EXPERIENCE_BUFFER_SIZE = 2
    GRID_SIZE              = 10 #the size of the FOV grid to apply to each agent
    ENVIRONMENT_SIZE       = (10,20)#the total size of the environment (length of one side)
    OBSTACLE_DENSITY       = (0,.2) #range of densities
    GRID_SIZE=torch.tensor(GRID_SIZE).share_memory_()
    ENVIRONMENT_SIZE=torch.tensor(ENVIRONMENT_SIZE).share_memory_()
    OBSTACLE_DENSITY=torch.tensor(OBSTACLE_DENSITY).share_memory_()

    DIAG_MVMT              = False # Diagonal movements allowed?
    a_size                 = 5 + int(DIAG_MVMT)*4
    SUMMARY_WINDOW         = 10
    NUM_META_AGENTS        = 1
    NUM_THREADS            = 2 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))

    NUM_BUFFERS            = 1 # NO EXPERIENCE REPLAY int(NUM_THREADS / 2)
    EPISODE_SAMPLES        = EXPERIENCE_BUFFER_SIZE # 64
    LR_Q                   = 2e-5  #8.e-5 / NUM_THREADS # default: 1e-5
    ADAPT_LR               = True
    ADAPT_COEFF            = 5.e-5 #the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
    load_model             = False
    RESET_TRAINER          = False
    model_path             = "pytorch_model"
    gifs_path              = 'gifs_primal'
    train_log_path         = 'train_log_path'

    # writer = SummaryWriter(log_dir=train_log_path)

    GLOBAL_NET_SCOPE       = 'global'

    #Imitation options
    PRIMING_LENGTH         = 0    # number of episodes at the beginning to train only on demonstrations
    DEMONSTRATION_PROB     = 0.2  # probability of training on a demonstration per episode

    # Simulation options
    FULL_HELP              = False
    OUTPUT_GIFS            = False
    SAVE_EPISODE_BUFFER    = False

    # Testing
    TRAINING               = True
    GREEDY                 = False
    NUM_EXPS               = 100
    MODEL_NUMBER           = 313000

    # Shared arrays for tensorboard
    episode_rewards        = [ [] for _ in range(NUM_META_AGENTS) ]
    episode_lengths        = [ [] for _ in range(NUM_META_AGENTS) ]
    episode_mean_values    = [ [] for _ in range(NUM_META_AGENTS) ]
    episode_invalid_ops    = [ [] for _ in range(NUM_META_AGENTS) ]
    episode_wrong_blocking = [ [] for _ in range(NUM_META_AGENTS) ]
    rollouts               = [ None for _ in range(NUM_META_AGENTS)]
    demon_probs=[np.random.rand() for _ in range(NUM_META_AGENTS)]
    # episode_steps_on_goal  = [ [] for _ in range(NUM_META_AGENTS) ]
    printQ                 = False # (for headless)
    swarm_reward           = [0]*NUM_META_AGENTS


    cpu_count=int(multiprocessing.cpu_count())
    print("CPU_Count=",cpu_count)
    print("number of Meta agents=",NUM_META_AGENTS)
    print("Threads=",NUM_THREADS)
    print("EXPERIENCE_BUFFER_SIZE=",EXPERIENCE_BUFFER_SIZE)


    print("Hello World")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
            
    
    manager = mp.Manager()
    shared_dict_obs = manager.dict()
    shared_dict_valid_act=manager.dict()
    state_dict=manager.dict()
    



    '''
        Observation: (position maps of current agent, current goal, other agents, other goals, obstacles)
            
        Action space: (Tuple)
            agent_id: positive integer
            action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
            5:NE, 6:SE, 7:SW, 8:NW}
        Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
    '''
    ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD,FINISH_REWARD,BLOCKING_COST = -0.3, -.5, 0.0, -2.,20.,-1.
    opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
    JOINT = False # True for joint estimation of rewards for closeby agents
    dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
    actionDict={v:k for k,v in dirDict.items()}



    '''
    State.
    Implemented as 2 2d numpy arrays.
    first one "state":
        static obstacle: -1
        empty: 0
        agent = positive integer (agent_id)
    second one "goals":
        agent goal = positive int(agent_id)
    '''
    def init_state(world0, goals, diagonal, num_agents=1):

        
        assert(len(world0.shape) == 2 and world0.shape==goals.shape)
        state_dict['state'] = torch.tensor(world0).share_memory_()
        state_dict['goals'] = torch.tensor(goals).share_memory_()
        state_dict['num_agents'] = torch.tensor(num_agents).share_memory_()
        agents, agents_past, agent_goals = scanForAgents(num_agents, state_dict['state'],state_dict['goals'])
        state_dict['agents'] = agents
        state_dict['agents_past'] = agents_past
        state_dict['agent_goals'] = agent_goals
        state_dict['diagonal'] = torch.tensor(diagonal).share_memory_()
        # print(state_dict['diagonal'])
        assert(len(agents) == num_agents)
        
        return state_dict 

    def scanForAgents(num_agents, state, goals):
        agents = [(-1,-1) for i in range(num_agents)]
        agents_last = [(-1,-1) for i in range(num_agents)]        
        agent_goals = [(-1,-1) for i in range(num_agents)]        
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if(state[i,j]>0):
                    agents[state[i,j]-1] = (i,j)
                    agents_last[state[i,j]-1] = (i,j)
                if(goals[i,j]>0):
                    agent_goals[goals[i,j]-1] = (i,j)
        assert((-1,-1) not in agents and (-1,-1) not in agent_goals)
        assert(agents==agents_last)
        agents= torch.tensor(agents).share_memory_()
        agents_last = torch.tensor(agents_last).share_memory_()
        agent_goals = torch.tensor(agent_goals).share_memory_()
        return agents, agents_last, agent_goals

    def getPos(agent_id, agents):
        # print(agents, agent_id,agents[agent_id-1])
        return agents[agent_id-1]

    def getPastPos(agent_id, agents_past):
        
        return agents_past[agent_id-1]

    def getGoal(agent_id, agent_goals):
        
        return agent_goals[agent_id-1]
        
    def diagonalCollision(agent_id, newPos, agents, num_agents, agents_past):
        '''diagonalCollision(id,(x,y)) returns true if agent with id "id" collided diagonally with 
        any other agent in the state after moving to coordinates (x,y)
        agent_id: id of the desired agent to check for
        newPos: coord the agent is trying to move to (and checking for collisions)
        '''
    #        def eq(f1,f2):return abs(f1-f2)<0.001
        def collide(a1,a2,b1,b2):
            '''
            a1,a2 are coords for agent 1, b1,b2 coords for agent 2, returns true if these collide diagonally
            '''
            return np.isclose( (a1[0]+a2[0]) /2. , (b1[0]+b2[0])/2. ) and np.isclose( (a1[1]+a2[1])/2. , (b1[1]+b2[1])/2. )
        assert(len(newPos) == 2)
        #up until now we haven't moved the agent, so getPos returns the "old" location
        lastPos = getPos(agent_id, agents)
        for agent in range(1,num_agents+1):
            if agent == agent_id: continue
            aPast = getPastPos(agent, agents_past)
            aPres = getPos(agent, agents)
            if collide(aPast,aPres,lastPos,newPos): return True
        return False

    #try to move agent and return the status
    def moveAgent(direction, agent_id, agents,agents_past, goals, state, diagonal):
        # with self.lock:
        axx=agents[agent_id-1][0] #tensor
        ayy=agents[agent_id-1][1] #tensor

        ax=axx.item()
        ay=ayy.item()

        # Not moving is always allowed
        if(direction==(0,0)):
            agents_past[agent_id-1]=agents[agent_id-1]
            return 1 if goals[ax,ay]==agent_id else 0

        # Otherwise, let's look at the validity of the move
        dx,dy =direction[0], direction[1]
        
        if(ax+dx>=state.shape[0] or ax+dx<0 or ay+dy>=state.shape[1] or ay+dy<0):#out of bounds
            return -1
        if(state[ax+dx,ay+dy]<0):#collide with static obstacle
            return -2
        if(state[ax+dx,ay+dy]>0):#collide with robot
            return -3
        # check for diagonal collisions
        if(diagonal):
            if diagonalCollision(agent_id,(ax+dx,ay+dy), agents,num_agents,agents_past):
                return -3
        # No collision: we can carry out the action
        state[ax,ay] = 0
        state[ax+dx,ay+dy] = agent_id

        # Save the previous position of the agent
        agents_past[agent_id-1]=agents[agent_id-1]
        # print( agents[agent_id-1])
        # agents[agent_id-1] = (ax+dx,ay+dy)
        # Update the agent's position in the tensor
        agents[agent_id-1, 0] = ax + dx  # Update x-coordinate 
        agents[agent_id-1, 1] = ay + dy  # Update y-coordinate

        # Check if the agent has reached its goal
        # print(goals," gap",len(goals)," gap",ax+dx," gap",ay+dy," gap",agent_id)
        if goals[ax+dx,ay+dy]==agent_id: #index out of errorbeacuse doing again ax+dx,ay+dy
            return 1
        elif goals[ax+dx,ay+dy]!=agent_id and goals[ax,ay]==agent_id:
            return 2
        else:
            return 0

        # try to execture action and return whether action was executed or not and why
        #returns:
        #     2: action executed and left goal
        #     1: action executed and reached goal (or stayed on)
        #     0: action executed
        #    -1: out of bounds
        #    -2: collision with wall
        #    -3: collision with robot
    def getDir(action):
        
        return dirDict[action]

    def act(action, agent_id, agents,agents_past, goals, state, diagonal):
        # 0     1  2  3  4 
        # still N  E  S  W
        direction = getDir(action)
        # direction=torch.tensor(direction).share_memory_()
        moved = moveAgent(direction,agent_id, agents,agents_past, goals, state, diagonal)
        return moved


    def getAction(direction):
        return actionDict[direction]

    # Compare with a plan to determine job completion
    def done(agents, goals):
        # with self.lock:
        numComplete = 0
        for i in range(1,len(agents)+1):
            agent_pos = agents[i-1]
            if goals[agent_pos[0],agent_pos[1]] == i:
                numComplete += 1
        return numComplete==len(agents) #, numComplete/float(len(self.agents))


    class MAPFEnv(gym.Env):
        def getFinishReward(self):
            return FINISH_REWARD
        metadata = {"render.modes": ["human", "ansi"]}

        # Initialize env
        # print("game ENV initialized")
        def __init__(self, num_agents=1, observation_size=10,world0=None, goals0=None, DIAGONAL_MOVEMENT=False, SIZE=(10,40), PROB=(0,.5), FULL_HELP=False,blank_world=False):
            """
            Args:
                DIAGONAL_MOVEMENT: if the agents are allowed to move diagonally
                SIZE: size of a side of the square grid
                PROB: range of probabilities that a given block is an obstacle
                FULL_HELP
            """
            # Initialize member variables
            self.num_agents        = num_agents 
            #a way of doing joint rewards
            self.individual_rewards           = [0 for i in range(num_agents)]
            self.observation_size  = observation_size
            self.SIZE              = SIZE
            self.PROB              = PROB
            self.fresh             = True
            self.FULL_HELP         = FULL_HELP
            self.finished          = False
            self.DIAGONAL_MOVEMENT = DIAGONAL_MOVEMENT

            # Initialize data structures
            self._setWorld(world0,goals0,blank_world=blank_world)
            if DIAGONAL_MOVEMENT:
                self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(9)])
            else:
                self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(5)])
            self.viewer           = None

        def isConnected(self,world0):
            sys.setrecursionlimit(10000)
            world0 = world0.copy()

            def firstFree(world0):
                for x in range(world0.shape[0]):
                    for y in range(world0.shape[1]):
                        if world0[x,y]==0:
                            return x,y
            def floodfill(world,i,j):
                sx,sy=world.shape[0],world.shape[1]
                if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                    return
                if(world[i,j]==-1):return
                world[i,j] = -1
                floodfill(world,i+1,j)
                floodfill(world,i,j+1)
                floodfill(world,i-1,j)
                floodfill(world,i,j-1)

            i,j = firstFree(world0)
            floodfill(world0,i,j)
            if np.any(world0==0):
                return False
            else:
                return True

        def getObstacleMap(self):
            return (self.world['state']==-1).to(torch.int)
        
        def getGoals(self):
            result=[]
            for i in range(1,self.num_agents+1):
                result.append(getGoal(i, self.world['agent_goals']))
            return result
        
        def getPositions(self):
            result=[]
            for i in range(1,self.num_agents+1):
                result.append(getPos(i, self.world['agents']))
            return result
        
        def _setWorld(self, world0=None, goals0=None,blank_world=False):
        
            #blank_world is a flag indicating that the world given has no agent or goal positions 
            def getConnectedRegion(world,regions_dict,x,y):
                sys.setrecursionlimit(1000000)
                '''returns a list of tuples of connected squares to the given tile
                this is memoized with a dict'''
                if (x,y) in regions_dict:
                    return regions_dict[(x,y)]
                visited=set()
                sx,sy=world.shape[0],world.shape[1]
                work_list=[(x,y)]
                while len(work_list)>0:
                    (i,j)=work_list.pop()
                    if(i<0 or i>=sx or j<0 or j>=sy):#out of bounds, return
                        continue
                    if(world[i,j]==-1):
                        continue#crashes
                    if world[i,j]>0:
                        regions_dict[(i,j)]=visited
                    if (i,j) in visited:continue
                    visited.add((i,j))
                    work_list.append((i+1,j))
                    work_list.append((i,j+1))
                    work_list.append((i-1,j))
                    work_list.append((i,j-1))
                regions_dict[(x,y)]=visited
                return visited
            #defines the State object, which includes initializing goals and agents
            #sets the world to world0 and goals, or if they are None randomizes world
            if not (world0 is None):
                print("not going in this condition")
                if goals0 is None and not blank_world:
                    print("not going in this condition")
                    raise Exception("you gave a world with no goals!")
                if blank_world:
                    #RANDOMIZE THE POSITIONS OF AGENTS
                    print("not going in this condition")
                    agent_counter = 1
                    agent_locations=[]
                    while agent_counter<=self.num_agents:
                        x,y       = np.random.randint(0,world0.shape[0]),np.random.randint(0,world0.shape[1])
                        if(world0[x,y] == 0):
                            world0[x,y]=agent_counter
                            agent_locations.append((x,y))
                            agent_counter += 1   
                    #RANDOMIZE THE GOALS OF AGENTS
                    goals0 = np.zeros(world0.shape).astype(int)
                    goal_counter = 1
                    agent_regions=dict()  
                    while goal_counter<=self.num_agents:
                        agent_pos=agent_locations[goal_counter-1]
                        valid_tiles=getConnectedRegion(world0,agent_regions,agent_pos[0],agent_pos[1])#crashes
                        x,y  = random.choice(list(valid_tiles))
                        if(goals0[x,y]==0 and world0[x,y]!=-1):
                            goals0[x,y]    = goal_counter
                            goal_counter += 1
                    self.initial_world = world0.copy()
                    self.initial_goals = goals0.copy()
                    self.world = init_state(self.initial_world,self.initial_goals,self.DIAGONAL_MOVEMENT,self.num_agents)
                    print("here")
                    return
                self.initial_world = world0
                self.initial_goals = goals0
                self.world = init_state(world0,goals0,self.DIAGONAL_MOVEMENT,self.num_agents)

                print("....................................inside sETWORL if condition.........................")
                return

            #otherwise we have to randomize the world
            print("below code is run always init and reset find out how agents go in mapf")
            #RANDOMIZE THE STATIC OBSTACLES
            prob=np.random.triangular(self.PROB[0],.33*self.PROB[0]+.66*self.PROB[1],self.PROB[1])
            size=np.random.choice([self.SIZE[0],self.SIZE[0]*.5+self.SIZE[1]*.5,self.SIZE[1]],p=[.5,.25,.25])
            world     = -(np.random.rand(int(size),int(size))<prob).astype(int)

            #RANDOMIZE THE POSITIONS OF AGENTS
            agent_counter = 1
            agent_locations=[]
            while agent_counter<=self.num_agents:
                x,y       = np.random.randint(0,world.shape[0]),np.random.randint(0,world.shape[1])
                if(world[x,y] == 0):
                    world[x,y]=agent_counter
                    agent_locations.append((x,y))
                    agent_counter += 1  
            # print("start position of mutiple  agents we are training=",agent_locations)    # to print the starting pos of current agent we are considering #farhan  
            
            #RANDOMIZE THE GOALS OF AGENTS
            goals = np.zeros(world.shape).astype(int)
            goal_counter = 1
            agent_regions=dict()     
            while goal_counter<=self.num_agents:
                agent_pos=agent_locations[goal_counter-1]
                valid_tiles=getConnectedRegion(world,agent_regions,agent_pos[0],agent_pos[1])
                x,y  = random.choice(list(valid_tiles))
                if(goals[x,y]==0 and world[x,y]!=-1):
                    goals[x,y]    = goal_counter
                    goal_counter += 1
            self.initial_world = world
            self.initial_goals = goals
            world1=world
            goals1=goals
            # world1=torch.tensor(world).share_memory_()
            # goals1=torch.tensor(goals).share_memory_()

            # Convert numpy arrays to shared memory
            # world,goals = store_the_reseted_world_to_share(world,goals)
            
            #convert back to numpy array
            # world =world.numpy()
            # goals =goals.numpy()

            # Initialize the shared State class
            print("new world created")
            self.world = init_state(world1, goals1,self.DIAGONAL_MOVEMENT,num_agents=self.num_agents)
            # print(self.world, self.world['agents'])
        


        # Returns an observation of an agent
        def _observe(self,agent_id):
            assert(agent_id>0)
            top_left=(getPos(agent_id, self.world['agents'])[0]-self.observation_size//2,getPos(agent_id, self.world['agents'])[1]-self.observation_size//2)
            bottom_right=(top_left[0]+self.observation_size,top_left[1]+self.observation_size)        
            obs_shape=(self.observation_size,self.observation_size)
            goal_map             = np.zeros(obs_shape)
            poss_map             = np.zeros(obs_shape)
            goals_map            = np.zeros(obs_shape)
            obs_map              = np.zeros(obs_shape)
            visible_agents=[]
            for i in range(top_left[0],top_left[0]+self.observation_size):
                for j in range(top_left[1],top_left[1]+self.observation_size):
                    if i>=self.world['state'].shape[0] or i<0 or j>=self.world['state'].shape[1] or j<0:
                        #out of bounds, just treat as an obstacle
                        obs_map[i-top_left[0],j-top_left[1]]=1
                        continue
                    if self.world['state'][i,j]==-1:
                        #obstacles
                        obs_map[i-top_left[0],j-top_left[1]]=1
                    if self.world['state'][i,j]==agent_id:
                        #agent's position
                        poss_map[i-top_left[0],j-top_left[1]]=1
                    if self.world['goals'][i,j]==agent_id:
                        #agent's goal
                        goal_map[i-top_left[0],j-top_left[1]]=1
                    if self.world['state'][i,j]>0 and self.world['state'][i,j]!=agent_id:
                        #other agents' positions
                        visible_agents.append(self.world['state'][i,j])
                        poss_map[i-top_left[0],j-top_left[1]]=1

            for agent in visible_agents:
                x, y = getGoal(agent, self.world['agent_goals'])
                min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                            max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
                goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

            dx=getGoal(agent_id, self.world['agent_goals'])[0]-getPos(agent_id, self.world['agents'])[0]  
            dy=getGoal(agent_id, self.world['agent_goals'])[1]-getPos(agent_id, self.world['agents'])[1]
            mag=(dx**2+dy**2)**.5 # magnitude for finidng unit vector to the goal 
            if mag!=0:
                dx=dx/mag
                dy=dy/mag

            
            
            poss_map=torch.tensor(poss_map).share_memory_()
            goal_map=torch.tensor(goal_map).share_memory_()
            goals_map=torch.tensor(goals_map).share_memory_()
            obs_map=torch.tensor(obs_map).share_memory_()
            dx= dx.share_memory_()
            dy= dy.share_memory_()
            mag=mag.share_memory_()
        
        
            shared_dict_obs[agent_id] = [[poss_map,goal_map,goals_map,obs_map],[dx,dy,mag]]
            return shared_dict_obs




        # Resets environment
        def _reset(self, agent_id,world0=None,goals0=None):
            self.finished = False
            # self.mutex.acquire()
            
            # Initialize data structures
            self._setWorld(world0,goals0)
            self.fresh = True
            
            # self.mutex.release()
            if self.viewer is not None:
                self.viewer = None
            on_goal = getPos(agent_id, self.world['agents']) == getGoal(agent_id, self.world['agent_goals'])
            #we assume you don't start blocking anyone (the probability of this happening is insanely low)
            return self._listNextValidActions(agent_id), on_goal,False

        def _complete(self):
            return done(self.world['agents'], self.world['goals'])
        
        def getAstarCosts(self,start, goal):
            #returns a numpy array of same dims as self.world['state'] with the distance to the goal from each coord
            def lowestF(fScore,openSet):
                #find entry in openSet with lowest fScore
                assert(len(openSet)>0)
                minF=2**31-1
                minNode=None
                for (i,j) in openSet:
                    if (i,j) not in fScore:continue
                    if fScore[(i,j)]<minF:
                        minF=fScore[(i,j)]
                        minNode=(i,j)
                return minNode      
            def getNeighbors(node):
                #return set of neighbors to the given node
                n_moves=9 if self.DIAGONAL_MOVEMENT else 5
                neighbors=set()
                for move in range(1,n_moves):#we dont want to include 0 or it will include itself
                    direction=getDir(move)
                    dx=direction[0]
                    dy=direction[1]
                    ax=node[0]
                    ay=node[1]
                    if(ax+dx>=self.world['state'].shape[0] or ax+dx<0 or ay+dy>=self.world['state'].shape[1] or ay+dy<0):#out of bounds
                        continue
                    if(self.world['state'][ax+dx,ay+dy]==-1):#collide with static obstacle
                        continue
                    neighbors.add((ax+dx,ay+dy))
                return neighbors
            
            #NOTE THAT WE REVERSE THE DIRECTION OF SEARCH SO THAT THE GSCORE WILL BE DISTANCE TO GOAL
            start,goal=goal,start
            
            # The set of nodes already evaluated
            closedSet = set()
        
            # The set of currently discovered nodes that are not evaluated yet.
            # Initially, only the start node is known.
            openSet =set()
            openSet.add(start)
        
            # For each node, which node it can most efficiently be reached from.
            # If a node can be reached from many nodes, cameFrom will eventually contain the
            # most efficient previous step.
            cameFrom = dict()
        
            # For each node, the cost of getting from the start node to that node.
            gScore =dict()#default value infinity
        
            # The cost of going from start to start is zero.
            gScore[start] = 0
        
            # For each node, the total cost of getting from the start node to the goal
            # by passing by that node. That value is partly known, partly heuristic.
            fScore = dict()#default infinity
        
            #our heuristic is euclidean distance to goal
            heuristic_cost_estimate = lambda x,y:math.hypot(x[0]-y[0],x[1]-y[1])
            
            # For the first node, that value is completely heuristic.        
            fScore[start] = heuristic_cost_estimate(start, goal)
        
            while len(openSet) != 0:
                #current = the node in openSet having the lowest fScore value
                current = lowestF(fScore,openSet)
        
                openSet.remove(current)
                closedSet.add(current)
                for neighbor in getNeighbors(current):
                    if neighbor in closedSet:
                        continue		# Ignore the neighbor which is already evaluated.
                
                    if neighbor not in openSet:	# Discover a new node
                        openSet.add(neighbor)
                    
                    # The distance from start to a neighbor
                    #in our case the distance between is always 1
                    tentative_gScore = gScore[current] + 1
                    if tentative_gScore >= gScore.get(neighbor,2**31-1):
                        continue		# This is not a better path.
                
                    # This path is the best until now. Record it!
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = gScore[neighbor] + heuristic_cost_estimate(neighbor, goal) 
        
            #parse through the gScores
            costs=self.world['state'].copy()
            for (i,j) in gScore:
                costs[i,j]=gScore[i,j]
            return costs
        
        def astar(self,obstacle_world,start,goal,robots=[]):
            '''robots is a list of robots to add to the world'''
            for (i,j) in robots:
                obstacle_world[i,j]=1
            try:
                path=cpp_mstar.find_path(obstacle_world,[start],[goal],1,5)
            except NoSolutionError:
                path=None
            for (i,j) in robots:
                obstacle_world[i,j]=0
            return path
        
        def get_blocking_reward(self,agent_id):
            '''calculates how many robots the agent is preventing from reaching goal
            and returns the necessary penalty'''
            print("checkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk already reached agent blocking other agent",agent_id)
            #accumulate visible robots
            other_robots=[]
            other_locations=[]
            inflation=3
            top_left1=(getPos(agent_id, self.world['agents'])[0]-self.observation_size//2,getPos(agent_id, self.world['agents'])[1]-self.observation_size//2)
            bottom_right1=(top_left1[0]+self.observation_size,top_left1[1]+self.observation_size)        
            for agent in range(1,self.num_agents):
                if agent==agent_id: continue
                x,y=getPos(agent, self.world['agents'])
                if x<top_left1[0] or x>=bottom_right1[0] or y>=bottom_right1[1] or y<top_left1[1]:
                    continue
                other_robots.append(agent)
                # other_locations.append((x,y))
                pos_tensor=getPos(agent, self.world['agents'])
                pos_tuple=tuple(pos_tensor.tolist())
                other_locations.append(pos_tuple)
            num_blocking=0
            tensor_obstacle_world=self.getObstacleMap()
            obstacle_world=tensor_obstacle_world.cpu().numpy()
            for agent in other_robots:
                pos_tensor=getPos(agent, self.world['agents'])
                pos_tuple=tuple(pos_tensor.tolist())
                other_locations.remove(pos_tuple)
                #before removing
                path_before=self.astar(obstacle_world,tuple(getPos(agent, self.world['agents']).tolist()),tuple(getGoal(agent, self.world['agent_goals']).tolist()),
                                    robots=other_locations+[tuple(getPos(agent_id, self.world['agents']).tolist())])
                #after removing
                # path for the other agent, this time excluding the evaluated agent from the list of obstacles
                path_after=self.astar(obstacle_world,tuple(getPos(agent, self.world['agents']).tolist()),tuple(getGoal(agent, self.world['agent_goals']).tolist()),
                                    robots=other_locations)
                other_locations.append(tuple(getPos(agent, self.world['agents']).tolist()))
                if (path_before is None and path_after is None):continue
                if (path_before is not None and path_after is None):continue
                if (path_before is None and path_after is not None)\
                    or len(path_before)>len(path_after)+inflation:
                    num_blocking+=1
            
            print("num_blocking*BLOCKING_COST=",num_blocking*BLOCKING_COST)
            
            return num_blocking*BLOCKING_COST
                
            
        # Executes an action by an agent
        def _step(self, action_input,episode=0):
            #episode is an optional variable which will be used on the reward discounting
            self.fresh = False
            n_actions = 9 if self.DIAGONAL_MOVEMENT else 5

            # Check action input
            assert len(action_input) == 2, 'Action input should be a tuple with the form (agent_id, action)'
            assert action_input[1] in range(n_actions), 'Invalid action'
            assert action_input[0] in range(1, self.num_agents+1)

            # Parse action input
            agent_id = action_input[0]
            action   = action_input[1]

            

            #get start location of agent
            agentStartLocation = getPos(agent_id, self.world['agents'])
            
            # Execute action & determine reward
            action_status = act(action,agent_id, self.world['agents'], self.world['agents_past'], self.world['goals'], self.world['state'], self.world['diagonal'])
            valid_action=action_status >=0
            #     2: action executed and left goal
            #     1: action executed and reached/stayed on goal
            #     0: action executed
            #    -1: out of bounds
            #    -2: collision with wall
            #    -3: collision with robot
            blocking=False
            
            if action==0:#staying still
                
                if action_status == 1:#stayed on goal
                    reward=GOAL_REWARD
                    
                    xb=self.get_blocking_reward(agent_id)
                    reward+=xb
                    if xb<0:
                        blocking=True
                elif action_status == 0:#stayed off goal
                    reward=IDLE_COST
            else:#moving
                if (action_status == 1): # reached goal
                    reward = GOAL_REWARD
                elif (action_status == -3 or action_status==-2 or action_status==-1): # collision
                    reward = COLLISION_REWARD
                elif (action_status == 2): #left goal
                    reward=ACTION_COST
                else:
                    reward=ACTION_COST
            self.individual_rewards[agent_id-1]=reward

            if JOINT:
                visible=[False for i in range(self.num_agents)]
                v=0
                #joint rewards based on proximity
                for agent in range(1,self.num_agents+1):
                    #tally up the visible agents
                    if agent==agent_id:
                        continue
                    top_left=(getPos(agent_id, self.world['agents'])[0]-self.observation_size//2, \
                            getPos(agent_id, self.world['agents'])[1]-self.observation_size//2)
                    pos=getPos(agent, self.world['agents'])
                    if pos[0]>=top_left[0] and pos[0]<top_left[0]+self.observation_size\
                        and pos[1]>=top_left[1] and pos[1]<top_left[1]+self.observation_size:
                            #if the agent is within the bounds for observation
                            v+=1
                            visible[agent-1]=True
                if v>0:
                    reward=self.individual_rewards[agent_id-1]/2
                    #set the reward to the joint reward if we are 
                    for i in range(self.num_agents):
                        if visible[i]:
                            reward+=self.individual_rewards[i]/(v*2)

            # Perform observation
            state = self._observe(agent_id) 

            # Done?
            doneVar = done(self.world['agents'], self.world['goals'])
            self.finished |= doneVar

            # next valid actions
            nextActions = self._listNextValidActions(agent_id, action,episode=episode)

            # on_goal estimation
            posy =  getPos(agent_id, self.world['agents'])
            goaly = getGoal(agent_id, self.world['agent_goals'])
            # print(posy, goaly, torch.equal(posy, goaly))
            on_goal = torch.equal(getPos(agent_id, self.world['agents']), getGoal(agent_id, self.world['agent_goals']))
            
            
        
            

            return state, reward, doneVar, nextActions, on_goal, blocking, valid_action,action_status

        def _listNextValidActions(self, agent_id, prev_action=0,episode=0):
            available_actions = [0] # staying still always allowed

            # Get current agent position
            agent_pos = getPos(agent_id, self.world['agents'])
            ax,ay     = agent_pos[0],agent_pos[1]
            n_moves   = 9 if self.DIAGONAL_MOVEMENT else 5

            for action in range(1,n_moves):
                direction = getDir(action)
                dx,dy     = direction[0],direction[1]
                if(ax+dx>=self.world['state'].shape[0] or ax+dx<0 or ay+dy>=self.world['state'].shape[1] or ay+dy<0):#out of bounds
                    continue
                if(self.world['state'][ax+dx,ay+dy]<0):#collide with static obstacle
                    continue
                if(self.world['state'][ax+dx,ay+dy]>0):#collide with robot
                    continue
                # check for diagonal collisions
                if(self.DIAGONAL_MOVEMENT):
                    if diagonalCollision(agent_id,(ax+dx,ay+dy), self.world['agents'], self.world['num_agents'], self.world['agents_past']):
                        continue          
                #otherwise we are ok to carry out the action
                available_actions.append(action)

            if opposite_actions[prev_action] in available_actions:
                available_actions.remove(opposite_actions[prev_action])
            
            shared_dict_valid_act[agent_id]=torch.tensor(available_actions).share_memory_()      
            return shared_dict_valid_act

    


    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        
    else:
        print("CUDA is not available.")

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    num_agents=2
    metaAgentID=0 # set 0 to avaoid index out of range error
    master_network=ACNet(a_size,False,GRID_SIZE).to(device) # Generate global network
    master_network.share_memory()
    


    optimizer =  SharedAdam(master_network.parameters(), lr=LR_Q)
    optimizer.share_memory()
    
    
    shared_env=MAPFEnv(num_agents,PROB=(.3,.5),SIZE=(10,11),DIAGONAL_MOVEMENT=False)
    
    episode_count=0
    episode_count=torch.tensor(episode_count,dtype=torch.float64).share_memory_()
    # episode_step_count=0
    # episode_step_count=torch.tensor(episode_step_count,dtype=torch.float64).share_memory_()
    lock = mp.Lock()
    barrier = mp.Barrier(num_agents)
    
    
    
    processes = []
    for agent_id in range(1,num_agents+1):
        print(f'worker {agent_id}: started')
        p = mp.Process(target=work, args=(agent_id,master_network,optimizer,metaAgentID,a_size,max_episode_length,gamma,shared_env,episode_count,lock,barrier))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    

if not TRAINING:
    print([np.mean(plan_durations), np.sqrt(np.var(plan_durations)), np.mean(np.asarray(plan_durations < max_episode_length, dtype=float))])




