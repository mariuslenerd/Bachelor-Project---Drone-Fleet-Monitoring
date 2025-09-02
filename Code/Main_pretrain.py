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

    T = 1000
    buffer_size = 6000

    
    # ----- Create a map -----
    
    x_size = 20
    y_size = 30
    
    Map = RegionMap(x_size, y_size)
    list_of_small_pertb = [[0,4], [13,5], [17,25]]
    list_of_big_pert = [[6,9],[17,25]]
    Map.initialize_importance_map(list_of_small_pertb, list_of_big_pert)
    
    # ----- Create a fleet -----
    
    drone_init_pos = [[2,4], [2,15], [13, 23], [5,5]]
    state_shape = [7]
    
    F = Monitoring_Fleet(buffer_size, state_shape, Map)
    F.add_drones(drone_init_pos)    
    
    #----- Create save folder paths -----
    
    np.set_printoptions(threshold=np.inf)
    
    current_folder = os.getcwd()
    save_folder = current_folder + "/Results"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)    
        
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")    
    
    current_results = save_folder + "/" + date_time + "_pretraining"
    if not os.path.exists(current_results):
        os.makedirs(current_results)    
    
    #----- Simultaion -----
    
    #F.pretrain(current_results, N_iter=30000, load_data='/Results/06_19_2024_20_28_04', plot_pretrained=True)
    
    #----- Continue training -----
    
    F.continue_training(current_results, lr=1e-4, N_iter=20000, load_data='/Results/06_19_2024_20_28_04', pretrained_folder="/Results/06_20_2024_12_38_13_pretraining/Saved_models/policy_network_4999.pt", plot_pretrained=True)    
     