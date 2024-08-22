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

def worker(shared_world, lock):
    with lock:
        # Perform operations on the shared world object
        print(f"Worker {mp.current_process().name} is using the shared world object.")
        # Example operation (replace with actual methods)
        # shared_world.some_method()

def main():
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
    num_agents             = 2

    gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE, 
                                    observation_size=GRID_SIZE,PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)

    world=gameEnv.world

    # Create a manager
    manager = mp.Manager()

    
    # Create a lock for synchronizing access
    lock = mp.Lock()

    # Create worker processes
    processes = []
    for _ in range(4):  # Create 4 worker processes
        p = mp.Process(target=worker, args=(shared_world, lock))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("Finished all processes.")


if __name__ == '__main__':
    mp.set_start_method('spawn')  # For compatibility in various environments
    main()