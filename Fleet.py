# -*- coding: utf-8 -*-
import numpy as np
import os 
from datetime import datetime
import matplotlib.pyplot as plt
import torch 

from Drone import Drone
from RegionMap import RegionMap
from Buffer import ReplayBuffer
from Neural_net import *

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class Monitoring_Fleet():
    
    def __init__(self, buffer_size, state_shape, Map, pretrained_folder=None):
        
        self.N = 0
        self.list_of_drones = []
        self.buffer_size = buffer_size
        
        self.Map = Map
        self.y_size = Map.y_size
        self.x_size = Map.x_size
        self.reward_matrix = (np.ones((self.x_size, self.y_size)))
        
        # ----- Shared info -----
        
        self.shared_info = {}
        
        #----- RL components -----
        
        self.alpha = 1e-4
        self.gamma = 0.9
        
        self.buffer = ReplayBuffer(buffer_size, state_shape)
        
        self.policy_network = NN_model(h_map=Map.x_size, w_map=Map.y_size, alpha=self.alpha, o_ch=1, ks=3, st=1, padd=1, dil=1)
        self.target_network = NN_model(h_map=Map.x_size, w_map=Map.y_size, alpha=self.alpha, o_ch=1, ks=3, st=1, padd=1, dil=1)     
        
        self.train_counter = 0
        
        self.log_avg_losses = []
        
        self.log_average_score = []
        
        #----- Load pretrained -----
        
        if pretrained_folder is not None:
            
            load_weights = os.getcwd() + pretrained_folder
            self.policy_network.load_checkpoint(load_weights)
            self.target_network.load_checkpoint(load_weights)
            
    def add_drones(self, drone_init_pos):
        
        self.N += len(drone_init_pos)
        
        for i in range(len(drone_init_pos)):
            
            x = drone_init_pos[i][0]
            y = drone_init_pos[i][1]
            self.list_of_drones.append(Drone(x,y))
                                       
    def update_reward_matrix_solo(self, drone):
        
        a = 0.3

        b = 0.9

        c = 0.01
        
        x = drone.x
        y = drone.y
        
        adjacent_nodes = [[x-1,y], [x+1,y], [x,y-1], [x,y+1]]
        
        self.reward_matrix[x, y] = self.reward_matrix[x, y]*a
        
        for node in adjacent_nodes:
            
            node_x = node[0]
            node_y = node[1]
            
            if 0 <= node_x <= self.Map.x_size-1 and 0 <= node_y <= self.Map.y_size-1:
                
                self.reward_matrix[node_x, node_y] *= b
               
        for x in range(self.x_size):

            for y in range(self.y_size):

                if self.reward_matrix[x,y]<1:

                    self.reward_matrix[x,y] = self.reward_matrix[x,y]+c

                elif self.reward_matrix[x,y]>1:

                    self.reward_matrix[x,y] = 1

    def get_drone_positions(self):
        
        list_of_drone_positions = []
        for drone in self.list_of_drones:
            list_of_drone_positions.append([drone.x, drone.y])
            
        return list_of_drone_positions
    
    def get_all_before(self, t, T):
        
        list_of_observations = []
        list_of_drone_positions = self.get_drone_positions()
        
        for idx, drone in enumerate(self.list_of_drones):
            
            d_x = drone.x
            d_y = drone.y
            d_observation = drone.get_observation(t, d_x, d_y, self.Map, self.reward_matrix, T, self.shared_info)

            list_of_observations.append(d_observation)
            
        return list_of_observations, list_of_drone_positions
    
    def get_all_after(self, t, T):
        
        list_of_observations_prime = []
        list_of_drone_positions_prime = self.get_drone_positions()
        
        for idx, drone in enumerate(self.list_of_drones):
            
            self.update_reward_matrix_solo(drone)
            
            d_x = drone.x
            d_y = drone.y
            d_observation_prime = drone.get_observation(t+1, d_x, d_y, self.Map, self.reward_matrix, T, self.shared_info)
            
            list_of_observations_prime.append(d_observation_prime)
            
            
        return list_of_observations_prime, list_of_drone_positions_prime 
    
    def get_state_(self, coordinates, list_of_observations):

         TMB = self.temporal_map_for_buffer(list_of_observations)
         TMB_Map = TMB.reshape(20, 30)
         mask_map = self.reward_matrix
         mask_map_flatten = mask_map.flatten()
         
         list_of_avg_masks = []
         
         for idx in range(len(list_of_observations)):
            
             drone_x = list_of_observations[idx]["x"]
             drone_y = list_of_observations[idx]["y"]
             
             avg_mask = [0.0, 0.0, 0.0, 0.0] #up, down, left, right, stay
             
             n = 0
             for index in range(0, drone_x):
                 n += 1
                 avg_mask[0] += mask_map[index, drone_y]
                 
             if n>0:    
                 avg_mask[0] = avg_mask[0]/n
             n = 0
                
             for index in range(drone_x, 20):
                 n += 1
                 avg_mask[1] += mask_map[index, drone_y]
             if n>0:    
                 avg_mask[1] = avg_mask[1]/n
             n = 0
             
             for index in range(0, drone_y):
                 n += 1                 
                 avg_mask[2] += mask_map[drone_x, index]
             if n>0:    
                 avg_mask[2] = avg_mask[2]/n
             n = 0
             
             for index in range(drone_y, 30):
                 n += 1                  
                 avg_mask[3] += mask_map[drone_x, index]
             if n>0:    
                 avg_mask[3] = avg_mask[3]/n
             n = 0
            
             list_of_avg_masks.append(avg_mask)
        
         coeff1 = 0.1
         coeff2 = 10.0

         s1 = list(np.array(coordinates[0:2])*coeff1) + list(np.array(list_of_observations[0]["temporal_importance_values"])*coeff2) + list_of_avg_masks[0]
         s2 = list(np.array(coordinates[2:4])*coeff1) + list(np.array(list_of_observations[1]["temporal_importance_values"])*coeff2) + list_of_avg_masks[1]
         s3 = list(np.array(coordinates[4:6])*coeff1) + list(np.array(list_of_observations[2]["temporal_importance_values"])*coeff2) + list_of_avg_masks[2]
         s4 = list(np.array(coordinates[6:8])*coeff1) + list(np.array(list_of_observations[3]["temporal_importance_values"])*coeff2) + list_of_avg_masks[3]

         S1 = np.array(s1)
         S2 = np.array(s2)
         S3 = np.array(s3)
         S4 = np.array(s4)

         return S1, S2, S3, S4, mask_map, mask_map_flatten, TMB, TMB_Map

    def get_reward_(self, t, T, S1, S2, S3, S4, S1_p, S2_p, S3_p, S4_p, mask_map_store, mask_map_updated, list_of_observations, list_of_observations_prime):
        
        mask_map_store = mask_map_store.reshape(20,30)
        mask_map_updated = mask_map_updated.reshape(20,30)
        
        coeff = 10.0
        R_list   = []
        
        alpha1 = 1.0
        alpha2 = 0.5
        
        for idx in range(len(list_of_observations)):
            
            adjacent_nodes = list_of_observations[idx]["adjacent_nodes"]
            temporal_importance_values = list_of_observations[idx]["temporal_importance_values"]

            adjacent_nodes_p = list_of_observations_prime[idx]["adjacent_nodes"]
            temporal_importance_values_p = list_of_observations_prime[idx]["temporal_importance_values"]
            
            previous = 0.0
            
            for (x,y), importance in zip(adjacent_nodes, temporal_importance_values):
                
                if importance > 0.0:
                    
                    previous += -importance
                    
            current = 0.0
            
            for (x_p,y_p), importance_p in zip(adjacent_nodes_p, temporal_importance_values_p):
                
                if importance_p > 0.0:
                    
                    current += importance_p 
                    
            #----- Add part of the reward that will encourage patrolling -----
            
            previous_mask = 0.0

            for (x,y), importance in zip(adjacent_nodes, temporal_importance_values):
                
                if importance != 0.0:
                    
                    previous_mask += -mask_map_store[x,y]
                    
            current_mask = 0.0
                
            for (x_p,y_p), importance_p in zip(adjacent_nodes_p, temporal_importance_values_p):
                
                if importance_p > 0.0:
                    
                    current_mask += mask_map_updated[x_p, y_p]                       
                    
            R_list.append(alpha1 *(current + previous) + alpha2*(current_mask + previous_mask))
                    
        return coeff*R_list[0], coeff*R_list[1], coeff*R_list[2], coeff*R_list[3]        

    def temporal_map_for_buffer(self,list_of_observations): 
        
        temporal_imp_buff = (np.ones((self.x_size+1,self.y_size+1)))*(-1.0)
        
        for observation in list_of_observations: 
            
            adjacent_nodes = observation['adjacent_nodes']
            temp_adj = observation['temporal_importance_values']
            
            for (x,y), importance in zip(adjacent_nodes, temp_adj): 
                
                temporal_imp_buff[x,y] = importance
            
        temporal_imp_buff = temporal_imp_buff [:-1, :-1]
        
        TMB = temporal_imp_buff.flatten()
        
        return TMB
    
    def plot_fleets_trajectories(self):
        
        fig, ax = plt.subplots()
        im = ax.imshow(self.Map.importance_map, cmap='viridis')
        list_of_markers = ["o", "v", "s", "p", "D", "8", "P", "X", "d", "4"]
    
        for idx, drone in enumerate(self.list_of_drones):
            
            positions = drone.history
            x_positions = [pos[0] for pos in positions]
            y_positions = [pos[1] for pos in positions]
            ax.plot(y_positions, x_positions)
            
            marker_idx = list_of_markers[idx]
            plt.scatter(drone.y, drone.x, color='black', facecolors='none',linewidth=1,marker=marker_idx,s=200)
            
        plt.xlabel("y-coordinate")
        plt.ylabel("x-coordinate")
        plt.title("Drone Trajectories on Importance Map")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Importance")
        
        plt.show()

#----- Different fleet agents -----
            
    def move_drones_random(self, list_of_observations, mask_map_store, TMB_Map):
        
        list_of_actions = [] # stores the actions for all drones
                     
        for idx, drone in enumerate(self.list_of_drones):
            
            d_observation = list_of_observations[idx]
            d_action      = drone.choose_action(d_observation, agent_type='random')
            list_of_actions.append(d_action)
            drone.update_drone(d_observation, d_action)
            
        return list_of_actions

    def move_drones_greedy(self, list_of_observations, mask_map_store, TMB_Map):
        
        list_of_actions = [] # stores the actions for all drones
                     
        for idx, drone in enumerate(self.list_of_drones):
            
            d_observation = list_of_observations[idx]
            d_action      = drone.choose_action(d_observation, agent_type='G')
            list_of_actions.append(d_action)
            drone.update_drone(d_observation, d_action)
            
        return list_of_actions
    
    def move_drones_SLS(self, list_of_observations, mask_map_store, TMB_Map):
        
        list_of_actions = [] # stores the actions for all drones
        
        agent_types = ['G', 'G', 'random', 'exp']
        
        for idx, drone in enumerate(self.list_of_drones):
            
            agent_type = agent_types[idx]
            d_observation = list_of_observations[idx]
            
            d_action = drone.choose_action(d_observation, agent_type)
            
            drone.update_drone(d_observation, d_action)
            list_of_actions.append(d_action)
            
            self.update_reward_matrix_solo(drone)
        
        return list_of_actions
    
    def move_drone_net(self, list_of_observations, mask, TMB):
        
        thrs = 0.5
        
        coeff1 = 0.1
        coeff2 = 10.0
        
        p = np.random.rand()
        
        if p < thrs:
            
            list_of_actions = self.move_drones_random(list_of_observations, mask, TMB)
            
        else:
        
            list_of_actions = []
            
            self.policy_network.eval()
            
            for idx, drone in enumerate(self.list_of_drones):

                drone_x = drone.x
                drone_y = drone.y
                 
                avg_mask = [0.0, 0.0, 0.0, 0.0] #up, down, left, right, stay
                 
                n = 0
                for index in range(0, drone_x):
                     n += 1
                     avg_mask[0] += mask[index, drone_y]
                     
                if n>0:    
                     avg_mask[0] = avg_mask[0]/n
                n = 0
                    
                for index in range(drone_x, 20):
                     n += 1
                     avg_mask[1] += mask[index, drone_y]
                if n>0:    
                     avg_mask[1] = avg_mask[1]/n
                n = 0
                 
                for index in range(0, drone_y):
                     n += 1                 
                     avg_mask[2] += mask[drone_x, index]
                if n>0:    
                     avg_mask[2] = avg_mask[2]/n
                n = 0
                 
                for index in range(drone_y, 30):
                     n += 1                  
                     avg_mask[3] += mask[drone_x, index]
                if n>0:    
                     avg_mask[3] = avg_mask[3]/n
                n = 0
                
                s = list(np.array([drone.x, drone.y])*coeff1) + list(np.array(list_of_observations[idx]["temporal_importance_values"])*coeff2) + avg_mask               
                s = np.array(s, dtype=np.float32)
                s = torch.tensor(s).float().unsqueeze(0).to(self.policy_network.device)
                
                q_values_torch = self.policy_network.forward(s).detach()
                q_values = q_values_torch.cpu().numpy().reshape(5)
                
                if idx == 0:
                    print(q_values)
                
                feasible_actions = drone.feasible(self.Map)
    
                q_value_max = - np.infty
                idx_max = - np.infty
                
                for idx_q, q_value in enumerate(q_values):
                    
                    if idx_q in feasible_actions:
                        
                        if q_value > q_value_max:
                            
                            idx_max = idx_q
                            q_value_max = q_value
                            
                list_of_actions.append(idx_max)
                
                drone.update_drone(list_of_observations[idx], list_of_actions[idx])
                                    
        return list_of_actions

#----- RL related components -----    
      
    def train(self, n_epochs, states_NN, actions_NN, rewards_NN, next_state_NN, online_train_lr=None):
        
        self.policy_network.train()
        
        if online_train_lr is not None:
            self.policy_network.optimizer.lr = online_train_lr
        
        #----- State -----

        state_batch  = states_NN
        
        reward_batch = rewards_NN
        
        action_batch = actions_NN
        
        state_batch_ = next_state_NN
        
        #----- Convert to tensor -----
        
        state_batch   = torch.tensor(state_batch ).float().to(self.policy_network.device)        
        reward_batch  = torch.tensor(reward_batch).float().to(self.policy_network.device)
        action_batch  = torch.tensor(action_batch).to(torch.int64).to(self.policy_network.device)
        state_batch_  = torch.tensor(state_batch_ ).float().to(self.policy_network.device)  
        
        #----- train -----
        
        list_of_losses_current = []
        
        for ep in range(n_epochs):
            
            state_action_values = self.policy_network(state_batch).gather(1, action_batch)
            
            with torch.no_grad():
                
                next_state_values = self.target_network(state_batch_).max(1)[0]
            
            expected_values = reward_batch + self.gamma * next_state_values
            
            expected_values = expected_values.unsqueeze(1)
            
            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_values)
             
            list_of_losses_current.append(loss.detach().cpu().numpy())
            
            #----- Clear gradients -----
            
            self.policy_network.optimizer.zero_grad()
            loss.backward()
            self.policy_network.optimizer.step()
            #self.policy_network.scheduler.step(loss)
            
            self.train_counter += 1
            
            if self.train_counter % (int(n_epochs/2)) == 0:

                self.target_network.load_state_dict(self.policy_network.state_dict())
            
        avg_loss = np.sum(np.array(list_of_losses_current))/n_epochs
        self.log_avg_losses.append(avg_loss)
        
        self.log_average_score.append(np.mean(self.log_avg_losses[-100:]))
               
    def pretrain(self, ckpt_file_name, N_iter=3000, load_data='/Results/06_14_2024_15_48_00', plot_pretrained=True):
        
        folder = os.getcwd() + load_data
        
        ckpt_file_name = ckpt_file_name + '/Saved_models'
        if not os.path.isdir(ckpt_file_name):
            os.makedirs(ckpt_file_name) 

        self.buffer.load(folder)
        
        print("Counter: ", self.buffer.mem_cntr)
        
        batch_size = 32
        n_epochs = 30
        
        for it in range(N_iter):
            
            print("Iteration:", it)
            
            states_NN, actions_NN, rewards_NN, next_state_NN = self.buffer.sample_buffer(batch_size)
            self.train(n_epochs, states_NN, actions_NN, rewards_NN, next_state_NN)
            
            if it % 500 == 499:
                
                self.policy_network.save_checkpoint(it, ckpt_file_name)

                if plot_pretrained:
            
                    fig1, ax1 = plt.subplots(dpi=180)
            
                    k = np.arange(len(self.log_average_score))
            
                    ax1.set_yscale('log')
                    ax1.plot(k, self.log_average_score)    
                    ax1.grid('on')
                    ax1.legend()
                    ax1.set_xlabel(r'$k$')
                    ax1.set_ylabel(r'$Loss$')
                    
                    fig1.savefig(ckpt_file_name + "/reward"+"_"+str(it)+".jpg", dpi=180)
                
            if it % 100 == 99:
                
                print("Loss: ", self.log_avg_losses[-1])
                
        if plot_pretrained:
            
            fig1, ax1 = plt.subplots(dpi=180)
            
            k = np.arange(len(self.log_average_score))
            
            ax1.set_yscale('log')
            ax1.plot(k, self.log_average_score)    
            ax1.grid('on')
            ax1.legend()
            ax1.set_xlabel(r'$k$')
            ax1.set_ylabel(r'$Loss$')    
            
            fig1.savefig(ckpt_file_name + "/reward.jpg", dpi=180)
            
        np.save(ckpt_file_name + "/log_average_score.npy", np.array(self.log_average_score))
        np.save(ckpt_file_name + "/log_avg_losses.npy"   , np.array(self.log_avg_losses))
        
        
    def continue_training(self, ckpt_file_name, lr=1e-4, N_iter=3000, load_data='/Results/06_19_2024_20_28_04', pretrained_folder="/Results/06_20_2024_01_31_10_pretraining/Saved_models/policy_network_29999.pt", plot_pretrained=True):
        
        #----- ckpt_file_name is the name of the folder where we save the results after continued training -----
        
        folder = os.getcwd() + load_data
        self.buffer.load(folder)
        
        ckpt_file_name = ckpt_file_name + '/Saved_models'
        if not os.path.isdir(ckpt_file_name):
            os.makedirs(ckpt_file_name) 

        self.policy_network = NN_model(h_map=self.x_size, w_map=self.y_size, alpha=lr, o_ch=1, ks=3, st=1, padd=1, dil=1)
        self.target_network = NN_model(h_map=self.x_size, w_map=self.y_size, alpha=lr, o_ch=1, ks=3, st=1, padd=1, dil=1)
            
        if pretrained_folder is not None:
            
            load_weights = os.getcwd() + pretrained_folder
            self.policy_network.load_checkpoint(load_weights)
            self.target_network.load_checkpoint(load_weights) 
            
        batch_size = 32
        n_epochs = 30
        
        for it in range(N_iter):
            
            print("Iteration:", it)
            
            states_NN, actions_NN, rewards_NN, next_state_NN = self.buffer.sample_buffer(batch_size)
            self.train(n_epochs, states_NN, actions_NN, rewards_NN, next_state_NN)
            
            if it % 500 == 499:
                
                self.policy_network.save_checkpoint(it, ckpt_file_name)

                if plot_pretrained:
            
                    fig1, ax1 = plt.subplots(dpi=180)
            
                    k = np.arange(len(self.log_average_score))
            
                    ax1.set_yscale('log')
                    ax1.plot(k, self.log_average_score)    
                    ax1.grid('on')
                    ax1.legend()
                    ax1.set_xlabel(r'$k$')
                    ax1.set_ylabel(r'$Loss$')
                    
                    fig1.savefig(ckpt_file_name + "/reward"+"_"+str(it)+".jpg", dpi=180)
                
            if it % 100 == 99:
                
                print("Loss: ", self.log_avg_losses[-1])
                
        if plot_pretrained:
            
            fig1, ax1 = plt.subplots(dpi=180)
            
            k = np.arange(len(self.log_average_score))
            
            ax1.set_yscale('log')
            ax1.plot(k, self.log_average_score)    
            ax1.grid('on')
            ax1.legend()
            ax1.set_xlabel(r'$k$')
            ax1.set_ylabel(r'$Loss$')    
            
            fig1.savefig(ckpt_file_name + "/reward.jpg", dpi=180)
            
        np.save(ckpt_file_name + "/log_average_score.npy", np.array(self.log_average_score))
        np.save(ckpt_file_name + "/log_avg_losses.npy"   , np.array(self.log_avg_losses))        
        
            
            
                
            

    

    
    
    
    
    
    
    