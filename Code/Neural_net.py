#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class NN_model(nn.Module):
    
    def __init__(self, h_map=20, w_map=30, o_ch=1, ks=3, st=1, padd=1, dil=1, alpha=0.001, ckpt_file_name=None):
    
        super(NN_model, self).__init__()
        
        self.h_map_init = h_map
        self.w_map_init = w_map
        
        #----- Extract features from the map -----
        
        self.conv1_map = nn.Conv2d(in_channels=1, out_channels=o_ch, kernel_size=ks, stride=st, padding=padd, dilation=dil)
        
        self.h_map  = np.floor((h_map + 2*padd - dil*(ks - 1) - 1)/st+1)
        self.w_map  = np.floor((w_map + 2*padd - dil*(ks - 1) - 1)/st+1)
        
        self.act1_map  = nn.ReLU()
        self.drop1_map = nn.Dropout(0.3)
        self.pool1_map = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.h_map_pad = np.floor((self.h_map + 2*0 - 1*(2 - 1) - 1)/2+1)
        self.w_map_pad = np.floor((self.w_map + 2*0 - 1*(2 - 1) - 1)/2+1)
        
        #----- Extract features from the mask -----
        
        self.conv2_mask = nn.Conv2d(in_channels=1, out_channels=o_ch, kernel_size=ks, stride=st, padding=padd, dilation=dil)
        
        self.h_mask  = self.h_map
        self.w_mask  = self.w_map 

        self.act2_mask  = nn.ReLU()
        self.drop2_mask = nn.Dropout(0.3)
        self.pool2_mask = nn.MaxPool2d(kernel_size=(2, 2))

        self.h_mask_pad = np.floor((self.h_map + 2*0 - 1*(2 - 1) - 1)/2+1)
        self.w_mask_pad = np.floor((self.w_map + 2*0 - 1*(2 - 1) - 1)/2+1)

        #----- Fully connected layers -----   
        
        self.feature_vector_size = int(o_ch*self.h_map_pad*self.w_map_pad + 0.0*o_ch*self.h_mask_pad*self.w_mask_pad + 2)
        
        self.bn0_state = nn.LayerNorm(7)
        
        self.fc3   = nn.Linear(11, 1024)
        #T.nn.init.xavier_uniform_(self.fc3.weight)
        self.act3  = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        
        
        self.fc4   = nn.Linear(1024, 512) 
        #T.nn.init.xavier_uniform_(self.fc4.weight)
        self.act4  = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)
        
        self.fc_int1   = nn.Linear(512, 256) 
        #T.nn.init.xavier_uniform_(self.fc4.weight)
        self.act_int1 = nn.ReLU()
        self.drop_int1= nn.Dropout(0.5)  
        
        self.fc_int2   = nn.Linear(256, 64) 
        #T.nn.init.xavier_uniform_(self.fc4.weight)
        self.act_int2 = nn.ReLU()
        self.drop_int2= nn.Dropout(0.5) 
        
        self.fc5 = nn.Linear(64,5)
        #T.nn.init.xavier_uniform_(self.fc5.weight)
        self.act5 = nn.ReLU()
        
        #----- Remaining parameters -----

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')       
        self.to(self.device)

        self.ckpt_file_name = ckpt_file_name
        
        #----- Put yourself on the appropriate device -----

        self.to(self.device)

    def forward(self, x):
        
        # map_  = T.reshape(map_ , (-1, 1, int(self.h_map), int(self.w_map)))
        # mask_ = T.reshape(mask_, (-1, 1, int(self.h_map), int(self.w_map)))
        
        # #----- Apply the first convolutional layer followed by ReLU activation, dropout and pooling on the map
        
        # map_  = self.act1_map(self.conv1_map(map_))
        # map_  = self.drop1_map(map_)
        # map_  = self.pool1_map(map_)   
        # map_vector = T.reshape(map_, (-1, int(self.h_map_pad*self.w_map_pad)))
        
        # #----- Apply the first convolutional layer followed by ReLU activation, dropout and pooling on the mask
        
        # mask_ = self.act2_mask(self.conv2_mask(mask_))
        # mask_ = self.drop2_mask(mask_)
        # mask_ = self.pool2_mask(mask_)
        # mask_vector = T.reshape(mask_, (-1, int(self.h_mask_pad*self.w_mask_pad)))

        #----- Apply the fully connected layers with ReLU activation and dropout
        
        # pos_ = T.reshape(pos_, (-1, 2))
        
        #print(map_vector.size(), mask_vector.size(), pos_.size())
        #x = T.cat((map_vector, mask_vector, pos_), axis=1)
        
        #x = T.cat((map_vector, pos_), axis=1)
        
        #x = self.bn0_state(x)
        
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        
        x = self.act4(self.fc4(x))
        x = self.drop4(x)
        
        x = self.act_int1(self.fc_int1(x))
        x = self.drop_int1(x) 

        x = self.act_int2(self.fc_int2(x))
        x = self.drop_int2(x) 
        
        x = self.act5(self.fc5(x))
        
        
        return x
    
    def save_checkpoint(self, iteration, ckpt_file_name):
        
        if not os.path.isdir(ckpt_file_name):
            os.makedirs(ckpt_file_name) 
            
        T.save(self.state_dict(), ckpt_file_name + '/policy_network_' + str(iteration) + '.pt')

        
    def load_checkpoint(self, ckpt_file_name):
        
        #print('... loading checkpoint ...')
        
        self.load_state_dict(T.load(ckpt_file_name))   

        
        
    

