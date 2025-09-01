# -*- coding: utf-8 -*-

from Drone import Drone
from RegionMap import RegionMap
from Buffer import ReplayBuffer
from Fleet import Monitoring_Fleet
from Neural_net import NN_model

import numpy as np
import os 
from datetime import datetime
import matplotlib.pyplot as plt
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

if __name__ == '__main__':

    T = 2000
    buffer_size = 6000
    batch_size = 64
    
    # ----- Create a map -----
    
    x_size = 20
    y_size = 30
    
    Map = RegionMap(x_size, y_size)
    list_of_small_pertb = [[0,4], [13,5], [17,25]]
    list_of_big_pert = [[6,9],[17,25]]
    Map.initialize_importance_map(list_of_small_pertb, list_of_big_pert)
    
    # ----- Create a fleet -----
    
    #drone_init_pos = [[1,1], [10, 1], [17, 1], [2,15], [11,15], [18, 15], [1,28], [9,28], [18, 28]]
    drone_init_pos = [[5,5], [5,20], [18, 5], [19,20]]
    state_shape    = [7+4]
    
    RL_active = True
    pretrained_folder = "/Results/06_20_2024_13_09_38_pretraining/Saved_models/policy_network_19999.pt"
    #pretrained_folder = None
    
    F = Monitoring_Fleet(buffer_size, state_shape, Map, pretrained_folder)
    F.add_drones(drone_init_pos)    
    
    #----- Create save folder paths -----
    
    np.set_printoptions(threshold=np.inf)
    
    current_folder = os.getcwd()
    save_folder = current_folder + "/Results"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)    
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")    
    
    current_results = save_folder + "/" + date_time
    if not os.path.exists(current_results):
        os.makedirs(current_results)     
    
    #----- Simultaion -----
    
    t_curr = 0
    
    while t_curr <= T:
        print('Time: ',t_curr)
        
    # ------------------------------------------------------------------------------------------
    
        list_of_observations, list_of_drone_positions = F.get_all_before(t_curr, T)
    
        list_of_drone_positions_flatten = []
        for x, y in list_of_drone_positions:
            list_of_drone_positions_flatten.append(x)
            list_of_drone_positions_flatten.append(y)
        
        if t_curr % 50 == 0:
            F.plot_fleets_trajectories() 
        
        S1, S2, S3, S4 , mask_map_store, mask_map_flatten, TMB, TMB_Map = F.get_state_(list_of_drone_positions_flatten, list_of_observations)
        
        #------------- MOVE -------------------------------------------------------------------------------------
        
        #list_of_actions = F.move_drones_random(list_of_observations, mask_map_store, TMB_Map)
        #list_of_actions = F.move_drones_greedy(list_of_observations, mask_map_store, TMB_Map)
        #list_of_actions = F.move_drones_SLS(list_of_observations, mask_map_store, TMB_Map)
        
        list_of_actions = F.move_drone_net(list_of_observations, mask_map_store, TMB_Map)

        #--------------------------------------------------------------------------------------------------------------------
        
        list_of_observations_prime, list_of_drone_positions_prime = F.get_all_after(t_curr, T)
       
        mask_map_updated = F.reward_matrix
        
        list_of_drone_positions_prime_flatten = []
        for x, y in list_of_drone_positions_prime:
            list_of_drone_positions_prime_flatten.append(x)
            list_of_drone_positions_prime_flatten.append(y)
        
        S1_p, S2_p, S3_p, S4_p, mask_map_updated, mask_map_updated_flatten, TMB_, TMB_Map_ = F.get_state_(list_of_drone_positions_prime_flatten, list_of_observations_prime)
        
        #----------------------------------------------------------------------------------------------------------------------
        
        R1, R2, R3, R4 = F.get_reward_(t_curr, T, S1, S2, S3, S4, S1_p, S2_p, S3_p, S4_p, mask_map_store, mask_map_updated, list_of_observations, list_of_observations_prime)
            
        states = [S1, S2, S3, S4]
        
        rewards = [R1, R2, R3, R4]

        new_states = [S1_p, S2_p, S3_p, S4_p]
        
        for state, state_, reward, action in zip(states, new_states, rewards, list_of_actions): 
            F.buffer.store_transition(state, action, reward, state_)
            
        for x in range(x_size):
            for y in range(y_size):
                Map.importance_map[x, y] = Map.dynamic_importance(t_curr, x, y, T)
        
        #---------- Sample for the NN : 64 * 673 = 43'072 value each time --------#
        
        if RL_active:
            
            n_epochs = 30
            
            states_NN, actions_NN, rewards_NN, next_state_NN = F.buffer.sample_buffer(batch_size)
            
            F.train(n_epochs, states_NN, actions_NN, rewards_NN, next_state_NN, online_train_lr=1e-4)
        
        t_curr += 1    
    
    F.buffer.save(current_results)
     