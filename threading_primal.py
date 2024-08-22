
from __future__ import division

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
import mapf_gym as mapf_gym
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
class Worker:
    def __init__(self,master_network, game, metaAgentID, workerID, a_size, groupLock):
        self.workerID = workerID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_"+str(workerID)
        self.agentID = ((workerID-1) % num_workers) + 1 
        self.groupLock = groupLock

        self.nextGIF = episode_count # For GIFs output
        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = ACNet(a_size,True,GRID_SIZE).to(device) #this should be our model?

        # print("Model's state_dict:")
        # for param_tensor in self.local_AC .state_dict():
        #     print(param_tensor, "\t", self.local_AC .state_dict()[param_tensor].size())

        # sync the weights at the beginning
        self.local_AC.load_state_dict(master_network.state_dict())
        # self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)
        

    def synchronize(self):
        #handy thing for keeping track of which to release and acquire
        if(not hasattr(self,"lock_bool")):
            self.lock_bool=False
        self.groupLock.release(int(self.lock_bool),self.name)
        self.groupLock.acquire(int(not self.lock_bool),self.name)
        self.lock_bool=not self.lock_bool
        
    def train(self, rollout, gamma, bootstrap_value, rnn_state0, imitation=False):
        global episode_count
        
        if ADAPT_LR:
        #computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        #we need the +1 so that lr at step 0 is defined
            lr = torch.div(LR_Q, torch.sqrt(torch.add(1., torch.mul(ADAPT_COEFF, episode_count))))
        else:
            lr=LR_Q
        wandb.log({"current_learning_rate ": lr})
        
        #look into foreach=False may because of detach() it should be true(default) for better performance
        optimizer = torch.optim.Adam(master_network.parameters(), lr=lr)
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

                    

                        self.local_AC.train()
                        output=self.local_AC(inputs,goal_pos,state_in=rnn_state0,training=True)
                    
                        self.policy, self.value, self.state_out ,self.state_in, self.blocking, self.on_goal,self.policy_sig,self.entropy=output
                        self.optimal_actions  = torch.tensor(optimal_actions_rollout).to(device)
                        self.optimal_actions_onehot = torch.nn.functional.one_hot(self.optimal_actions, num_classes=a_size)
                        self.optimal_actions_onehot =torch.argmax(self.optimal_actions_onehot, dim=1)
                        self.optimal_actions_onehot=self.optimal_actions_onehot.long()

                        loss_function=nn.CrossEntropyLoss()
                        self.imitation_loss = loss_function(self.entropy,self.optimal_actions_onehot)
                        #cant do this directly because of the softmaxissue
                        # self.imitation_loss = torch.sum(self.optimal_actions_onehot*self.policy)

                        
                        
                        # Backward pass and optimization
                        optimizer.zero_grad()
                        self.imitation_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.local_AC.parameters(),max_norm=0.1)
                        # Manually push the gradients to the global network
                        ensure_shared_grads(self.local_AC,master_network) 
                        optimizer.step()
                        self.local_AC.load_state_dict(master_network.state_dict())

                        return self.imitation_loss
                        
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
                    
                global_step=episode_count

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
                
                self.rewards_plus = np.asarray(rewards1.tolist() + [bootstrap_value])
                if isinstance(self.rewards_plus, np.ndarray) and any(x is None for x in self.rewards_plus):
                    print("array1 contains None")
                discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
                # discounted_rewards= torch.stack([torch.tensor(item[-2]) for item in rollout])
                value_temp=[tensor.cpu().detach().numpy() for tensor in values_rollout]
                values1=np.array(value_temp,dtype=object)
                self.value_plus = np.asarray(values1.tolist() + [bootstrap_value])
                # values_tensor = torch.tensor(values.tolist(), dtype=torch.float32)
                # self.value_plus = values_tensor + bootstrap_value_tensor
                advantages = rewards1 + gamma * self.value_plus[1:] - self.value_plus[:-1]
                advantages1 = good_discount(advantages,gamma)


                num_samples = min(EPISODE_SAMPLES,len(advantages1))
                sampleInd = np.sort(np.random.choice(advantages1.shape[0], size=(num_samples,), replace=False))
                advantages2=np.array(advantages1, dtype=np.float32)

                
                # observations = torch.stack(observations).to(device)
                # goals=torch.stack(goals_rollout).to(device)
                self.local_AC.train()
                inputs1=torch.tensor(observations1,dtype=torch.float32).to(device)
                goal_pos1=torch.tensor(goals1,dtype=torch.float32).to(device)

                # Monitor initial memory usage
                # print(f"Initial allocated memory: {torch.cuda.memory_allocated()} bytes")
                # print(f"Initial reserved memory: {torch.cuda.memory_reserved()} bytes")


                output=self.local_AC(inputs1,goal_pos1,state_in=rnn_state0,training=True)


                # Monitor memory usage after forward pass
                # print(f"After forward pass allocated memory: {torch.cuda.memory_allocated()} bytes")
                # print(f"After forward pass reserved memory: {torch.cuda.memory_reserved()} bytes")

                self.policy, self.value, self.state_out ,self.state_in, self.blocking, self.on_goal,self.policy_sig,self.entropy=output
                


                self.target_v   = torch.from_numpy(discounted_rewards.copy()).to(device)  #torch.tensor(discounted_rewards)
                self.actions  = torch.tensor(actions_rollout).to(device)

                self.actions_onehot = torch.nn.functional.one_hot(self.actions, num_classes=a_size)
                self.train_valid = torch.stack(valids_rollout).to(device)
                self.advantages= torch.from_numpy(advantages2.copy()).to(device) #torch.tensor(advantages)
                self.train_value=torch.stack(train_value_rollout).to(device)
                self.target_blockings=torch.stack(blockings_rollout).to(device)
                self.target_on_goals =torch.stack(on_goals_rollout).to(device)
                self.responsible_outputs    = torch.sum(self.policy * self.actions_onehot,dim=1)
                length_roll=len(observations_rollout) #length of the rollouts
                

                #calculate loss
                self.value_loss    = torch.sum(self.train_value * torch.square((self.target_v-self.value.view(-1))))
                self.entropy       = - torch.sum(self.policy*(torch.log((torch.clamp(self.policy, min=1e-10, max=1.0)))))
                self.policy_loss   = torch.sum(self.advantages*(torch.log((torch.clamp(self.responsible_outputs, min=1e-15, max=1.0)))))
                self.valid_loss    = - torch.sum(((torch.log((torch.clamp(self.policy_sig, min=1e-10, max=1.0))))*self.train_valid)+((torch.log(torch.clamp(1-self.policy_sig, min=1e-10, max=1.0)))*(1-self.train_valid)))
                self.blocking_loss = - torch.sum(((torch.log((torch.clamp(self.blocking, min=1e-15, max=1.0))))*self.target_blockings)+((torch.log((torch.clamp(1-self.blocking, min=1e-15, max=1.0))))*(1-self.target_blockings)))
                self.on_goal_loss  = - torch.sum(((torch.log((torch.clamp(self.on_goal, min=1e-10, max=1.0))))*self.target_on_goals)+((torch.log((torch.clamp(1-self.on_goal, min=1e-10, max=1.0))))*(1-self.target_on_goals)))
                self.loss          = (0.5 * self.value_loss + self.policy_loss + 0.5*self.valid_loss - self.entropy * 0.01 +.5*self.blocking_loss+0.5*self.on_goal_loss).to(device)
                
                #calculated loss and apply gradient

                optimizer.zero_grad()
                # print(torch.cuda.max_memory_allocated())
                # print(torch.cuda.memory_allocated())
                # print(torch.cuda.memory_summary())

                self.loss.backward()
                # Print parameters that have None gradients
                for name, param in self.local_AC.named_parameters():
                    if param.grad is None:
                        print(f"Parameter with None gradient: {name}")

                #  check for gradients
                # for param in self.local_AC.parameters():
                    # if param.grad is not None:
                        # print(f"Param: {param.shape}, Grad type: {type(param.grad)}, Grad shape: {param.grad.shape}")



                # Monitor memory usage after backward pass
                # print(f"After backward pass allocated memory: {torch.cuda.memory_allocated()} bytes")
                # print(f"After backward pass reserved memory: {torch.cuda.memory_reserved()} bytes")


                torch.nn.utils.clip_grad_norm_(self.local_AC.parameters(),max_norm=1.0)
                # Manually push the gradients to the global network
                # print(self.agentID)
                ensure_shared_grads(self.local_AC,master_network)  
                
                    
                optimizer.step()
                 #sync the weights again
                self.local_AC.load_state_dict(master_network.state_dict())
                g_n=1
                v_n=1
                

                # Update the global network using gradients from loss
                # Generate network statistics to periodically save

                
                return self.value_loss/length_roll, self.policy_loss /length_roll, self.valid_loss/length_roll, self.entropy/length_roll, self.blocking_loss/length_roll, self.on_goal_loss/length_roll, g_n, v_n
                

    def shouldRun(self, stop_event, episode_count):
        if TRAINING:
            return not stop_event.is_set()
        else:
            return (episode_count < NUM_EXPS)

    def parse_path(self,path):
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
                observations.append(self.env._observe(agent))
            steps=0
            while len(move_queue)>0:
                steps+=1
                i=move_queue.pop(0)
                o=observations[i]
                pos=path[t][i]
                newPos=path[t+1][i]#guaranteed to be in bounds by loop guard
                direction=(newPos[0]-pos[0],newPos[1]-pos[1])
                a=self.env.world.getAction(direction)
                state, reward, done, nextActions, on_goal, blocking, valid_action=self.env._step((i+1,a))
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
        
    def work(self,max_episode_length,gamma,stop_event):
        global episode_count, swarm_reward, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops,episode_wrong_blocking #, episode_invalid_goals
        total_steps, i_buf = 0, 0
        episode_buffers, s1Values = [ [] for _ in range(NUM_BUFFERS) ], [ [] for _ in range(NUM_BUFFERS) ]
        
    
        while episode_count<60000:
            torch.autograd.set_detect_anomaly(True)
             #should implement the cordinator of thread
            
            # sync the weights at the beginning
            self.local_AC.load_state_dict(master_network.state_dict())
            
            episode_buffer, episode_values = [], []
            episode_reward = episode_step_count = episode_inv_count = 0
            d = False

            # Initial state from the environment
            if self.agentID==1:
                self.env._reset(self.agentID)
            self.synchronize() # synchronize starting time of the threads
            validActions          = self.env._listNextValidActions(self.agentID)
            s                     = self.env._observe(self.agentID)
            blocking              = False
            p=self.env.world.getPos(self.agentID)
            
            on_goal               = self.env.world.goals[p[0],p[1]]==self.agentID
            s                     = self.env._observe(self.agentID)
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

            if self.agentID==1:
                global demon_probs
                demon_probs[self.metaAgentID]=np.random.rand()
            self.synchronize() # synchronize starting time of the threads

            # reset swarm_reward (for tensorboard)
            swarm_reward[self.metaAgentID] = 0
            if episode_count<PRIMING_LENGTH and demon_probs[self.metaAgentID]<DEMONSTRATION_PROB:
                #for the first PRIMING_LENGTH episodes, or with a certain probability
                #don't train on the episode and instead observe a demonstration from M*
                if self.workerID==1 and episode_count%1000==0:
                    # Save the model state dictionary
                    torch.save({'model_state_dict': master_network.state_dict()}, f"{model_path}/model-{episode_count}.pt") 
                    torch.save({'model_state_dict': self.local_AC.state_dict()}, f"{model_path}/model-{episode_count+1}.pt")# local or global? should be global right?

                    print(f"Model saved at {model_path}/model-{episode_count}.pt")
                    # saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                    
                global rollouts
                rollouts[self.metaAgentID]=None
                if(self.agentID==1):
                    world=self.env.getObstacleMap()
                    start_positions=tuple(self.env.getPositions())
                    goals=tuple(self.env.getGoals())
                    try:
                        print("before")
                        mstar_path=cpp_mstar.find_path(world,start_positions,goals,2,5)
                        print("after")
                        rollouts[self.metaAgentID]=self.parse_path(mstar_path)
                    except OutOfTimeError:
                        #M* timed out 
                        print("timeout",episode_count)
                    except NoSolutionError:
                        print("nosol????",episode_count,start_positions)
                self.synchronize()
                if rollouts[self.metaAgentID] is not None:
                    i_l=self.train(rollouts[self.metaAgentID][self.agentID-1], gamma, None, rnn_state0, imitation=True)

                    episode_count+=1./num_workers
                    if self.agentID==1:
                        # print("imitation loss wrote to the summary")
                        # Log the imitation loss with the tag 'Losses/Imitation loss'
                        wandb.log({"Losses/Imitation_loss ": i_l})
                        # writer.add_scalar('Losses/Imitation_loss', i_l, episode_count)
                        # writer.flush()
                        # writer.close()
                        
                    continue
                continue
            saveGIF = False
            if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                saveGIF = True
                self.nextGIF =episode_count + 64
                GIF_episode = int(episode_count)
                episode_frames = [ self.env._render(mode='rgb_array',screen_height=900,screen_width=900) ]
                
            while (not self.env.finished): # Give me something!
                #Take an action using probabilities from policy network output.
                inputs=s[0] #observation #pos,goal,obs maps
                inputs=np.array(inputs)
                inputs=torch.tensor(inputs, dtype=torch.float32).to(device)
                inputs=inputs.unsqueeze(0) #accodomodate batchsize

                goal_pos=s[1] #dx,dy,mag
                goal_pos=np.array(goal_pos)
                goal_pos=torch.tensor(goal_pos, dtype=torch.float32).to(device)
                goal_pos=goal_pos.unsqueeze(0)
                

                self.local_AC.eval() # Set model to evaluation mode
                with torch.no_grad():
                
                    a_dist, v, rnn_state,state_in,pred_blocking,pred_on_goal,policy_sig,entropy=self.local_AC(inputs=inputs,goal_pos=goal_pos,state_in=state_init,training=True)
                #there is an other way using no_grad() instead of detach
                


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
                valid_dist = a_dist[0, validActions]
                valid_dist /= torch.sum(valid_dist)

                if TRAINING:
                    if (pred_blocking.flatten()[0] < 0.5) == blocking:
                        wrong_blocking += 1
                    if (pred_on_goal.flatten()[0] < 0.5) == on_goal:
                        wrong_on_goal += 1
                    a           = a = validActions[torch.multinomial(valid_dist, 1, replacement=True).item()]
                    train_val   = 1.
                else:
                    a         = torch.argmax(a_dist.flatten())
                    if a not in validActions or not GREEDY:
                        a     = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
                    train_val = 1.

                _, r, _, _, on_goal,blocking,_ = self.env._step((self.agentID, a),episode=episode_count)



                

                self.synchronize() # synchronize threads

                # Get common observation for all agents after all individual actions have been performed
                s1           = self.env._observe(self.agentID)
                validActions = self.env._listNextValidActions(self.agentID, a,episode=episode_count)
                d            = self.env.finished

                if saveGIF:
                    episode_frames.append(self.env._render(mode='rgb_array',screen_width=900,screen_height=900))

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
                        _,s1Values[i_buf],_,_,_,_,_,_=self.local_AC(inputs=inputs,goal_pos=goal_pos,state_in=rnn_state,training=True)
                        s1Values[i_buf]=s1Values[i_buf][0]

                        


                    if (episode_count-EPISODE_START) < NUM_BUFFERS:
                        i_rand = np.random.randint(i_buf+1)
                    else:
                        
                        
                        i_rand = np.random.randint(NUM_BUFFERS)
                
                        #detach each element
                        #put together again
                        # tmp = np.array(episode_buffers[i_rand],dtype=object)
                        
                        # tmp = np.array(episode_buffers[i_rand],dtype=object)

                        # tmp=torch.sta(episode_buffers[i_rand])
                        # while tmp.shape[0] == 0:
                            # i_rand = np.random.randint(NUM_BUFFERS)
                            # tmp = np.array(episode_buffers[i_rand])
                    v_l,p_l,valid_l,e_l,b_l,og_l,g_n,v_n = self.train(episode_buffers[i_rand],gamma,s1Values[i_rand],rnn_state0)

                    i_buf = (i_buf + 1) % NUM_BUFFERS
                    rnn_state0             = (rnn_state[0].clone().detach(),rnn_state[1].clone().detach())
                    episode_buffers[i_buf] = []

                self.synchronize() # synchronize threads
                
                # sync the weights at the end also???
    
                self.local_AC.load_state_dict(master_network.state_dict())

                # sess.run(self.pull_global)
                if episode_step_count >= max_episode_length or d:
                    break

            episode_lengths[self.metaAgentID].append(episode_step_count)
            episode_values=[t.detach() for t in episode_values]

            # episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
            episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
            episode_wrong_blocking[self.metaAgentID].append(wrong_blocking)

            # Periodically save gifs of episodes, model parameters, and summary statistics.
            if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                print('                                                                                   ', end='\r')
                print('{} Episode terminated ({},{})'.format(episode_count, self.agentID, RewardNb), end='\r')

            swarm_reward[self.metaAgentID] += episode_reward

            self.synchronize() # synchronize threads

            episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])

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
                episode_count+=1./num_workers
                if episode_count % 1000 == 0:
                        print("here")
                        print ('Saving Model', end='\n')
                        torch.save({'model_state_dict': master_network.state_dict()}, f"{model_path}/model-{episode_count}.pt") # local or global? should be global right?
                        torch.save({'model_state_dict': self.local_AC.state_dict()}, f"{model_path}/model-{episode_count+1}.pt")
                        print ('Saved Model', end='\n')
                        
                if episode_count % SUMMARY_WINDOW == 0:
                    if episode_count % 1000 == 0:
                        print("here")
                        print ('Saving Model', end='\n')
                        torch.save({'model_state_dict': master_network.state_dict()}, f"{model_path}/model-{episode_count}.pt") # local or global? should be global right?
                        torch.save({'model_state_dict': self.local_AC.state_dict()}, f"{model_path}/model-{episode_count+1}.pt")
                        print ('Saved Model', end='\n')
                    SL = SUMMARY_WINDOW * num_workers
                    mean_reward = np.nanmean(episode_rewards[self.metaAgentID][-SL:])
                    mean_length = np.nanmean(episode_lengths[self.metaAgentID][-SL:])
                    # mean_value = np.nanmean(episode_mean_values[self.metaAgentID][-SL:])
                    mean_invalid = np.nanmean(episode_invalid_ops[self.metaAgentID][-SL:])
                    mean_wrong_blocking = np.nanmean(episode_wrong_blocking[self.metaAgentID][-SL:])
                    # current_learning_rate = sess.run(lr,feed_dict={global_step:episode_count}) calculate current learning rate

                
                    # writer.add_scalar('Perf/Learning Rate', current_learning_rate,episode_count)
                    wandb.log({"Perf/Reward": mean_reward, "episode_count": episode_count})
                    wandb.log({"current_learning_rate ": lr})
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


# ## Training


# Learning parameters
max_episode_length     = 256
episode_count          = 0
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = 50
GRID_SIZE              = 10 #the size of the FOV grid to apply to each agent
ENVIRONMENT_SIZE       = (10,20)#the total size of the environment (length of one side)
OBSTACLE_DENSITY       = (0,.2) #range of densities
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


if not TRAINING:
    plan_durations = np.array([0 for _ in range(NUM_EXPS)])
    mutex = threading.Lock()
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        os.makedirs('gifs3D')

#Create a directory to save episode playback gifs to
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

#from here
    


    
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    
else:
    print("CUDA is not available.")

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with device:

    # torch.cuda.memory_snapshot()
    torch.cuda.empty_cache() #for freeing the cached memory from gpu
    # print(torch.cuda.max_memory_allocated())
    # print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_summary())


    master_network=ACNet(a_size,False,GRID_SIZE).to(device) # Generate global network

    stop_event = threading.Event()
    
    
    global_step = 0

    if ADAPT_LR:
        #computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        #we need the +1 so that lr at step 0 is defined
        lr = torch.div(LR_Q, torch.sqrt(torch.add(1., torch.mul(ADAPT_COEFF, global_step))))
    else:
        lr = torch.full((1,), LR_Q)

    trainer = torch.optim.NAdam(master_network.parameters(), lr=lr)
    # trainer.share_memory()

    if TRAINING:
        num_workers = NUM_THREADS # Set workers # = # of available CPU threads
        print("no of workers=",num_workers)
    else:
        num_workers = NUM_THREADS
        NUM_META_AGENTS = 1
    
    gameEnvs, workers, groupLocks = [], [], []
    n=1#counter of total number of agents (for naming)
    for ma in range(NUM_META_AGENTS):
        num_agents=NUM_THREADS # for 1  meta agent have agents equal to number of threads
        gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE, 
                                   observation_size=GRID_SIZE,PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)
        gameEnvs.append(gameEnv)

        # # Create groupLock
        workerNames = ["worker_"+str(i) for i in range(n,n+num_workers)] # identify each agent realted to which meta agent and grouped with lock
        groupLock = GroupLock.GroupLock([workerNames,workerNames])       #for synchronization of those agents
        groupLocks.append(groupLock)

        # # Create worker classes
        workersTmp = []
        for i in range(ma*num_workers+1,(ma+1)*num_workers+1):
            workersTmp.append(Worker(master_network,gameEnv,ma,n,a_size,groupLock))
            n+=1
        workers.append(workersTmp)

    # global_summary = tf.compat.v1.summary.FileWriter(train_path) # train_primal for visualization of tensorboard
    

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
    worker_threads = []
    # processes = []
        
    for ma in range(NUM_META_AGENTS):
        for worker in workers[ma]:
            worker_work = lambda: worker.work(max_episode_length,gamma,stop_event)
            # p = mp.Process(target=worker_work)
            # print("Starting worker " + str(worker.workerID))
            # p.start()
            # processes.append(p)
            groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
            # worker_work = worker.work(max_episode_length,gamma,stop_event )
            print("Starting worker " + str(worker.workerID))
            t = threading.Thread(target=(worker_work))
            t.start()
            worker_threads.append(t)
    # for p in processes:
        # p.join()
    
    for t in worker_threads:
        t.join()

if not TRAINING:
    print([np.mean(plan_durations), np.sqrt(np.var(plan_durations)), np.mean(np.asarray(plan_durations < max_episode_length, dtype=float))])




