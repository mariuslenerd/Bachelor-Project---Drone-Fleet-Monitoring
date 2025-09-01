# -*- coding: utf-8 -*-

import numpy as np
from RegionMap import RegionMap
from Drone import Drone
import pandas as pd

class ReplayBuffer():
    
    def __init__(self, max_size, state_shape):
        
        #----- Max mem size and current occupancy -----
        
        self.mem_size = max_size
        self.mem_cntr = 0
        
        n_actions = 1
        
        self.state_memory      = np.zeros((self.mem_size, *state_shape))
        self.next_state_memory = np.zeros((self.mem_size, *state_shape))
        self.action_memory     = np.zeros((self.mem_size, n_actions))
        self.reward_memory     = np.zeros(self.mem_size)
        
        self.total_buffer_array = np.zeros((self.mem_size, *state_shape))
        
    def store_transition(self, state, action ,reward, state_):
        
        if (reward >= 0.0 or reward <= -0.0) and np.abs(reward)>0.0:
        
            if self.mem_cntr < self.mem_size:
                
                index = self.mem_cntr
                
            else:
                
                index = self.mem_cntr % self.mem_size
            
            #----- size = 1202 2(20x30)+2 : (x,y) coordinates
            self.state_memory[index]      = state
            
            #----- size = 1        
            self.action_memory[index]     = action
            
            #----- size = 1 
            self.reward_memory[index]     = reward
            
            #----- size = 1202
            self.next_state_memory[index] = state_
            
            self.mem_cntr += 1
 
    def save(self, destination):
        
       #----- Convert the list of arrays to a DataFrame
       
       #df_state_memory      = pd.DataFrame(self.state_memory)
       #df_next_state_memory = pd.DataFrame(self.next_state_memory)
       #df_action_memory     = pd.DataFrame(self.action_memory)
       #df_reward_memory     = pd.DataFrame(self.reward_memory)
       
       #----- Save the DataFrame to a CSV file
       
       #df_state_memory.to_csv(destination + '/df_state_memory.csv', index=False, header=False)
       #df_next_state_memory.to_csv(destination + '/df_next_state_memory.csv', index=False, header=False)
       #df_action_memory.to_csv(destination + '/df_action_memory.csv', index=False, header=False)
       #df_reward_memory.to_csv(destination + '/df_reward_memory.csv', index=False, header=False)

       #----- Save numpy arrays
        
       np.save(destination + '/state_memory.npy', self.state_memory)
       np.save(destination + '/next_state_memory.npy', self.next_state_memory)
       np.save(destination + '/action_memory.npy', self.action_memory)
       np.save(destination + '/reward_memory.npy', self.reward_memory)
       
       np.save(destination + '/mem_cntr.npy', np.array(self.mem_cntr))
    
    def load(self, destination): 
        
        self.mem_cntr = int(np.load(destination + '/mem_cntr.npy'))
        
        self.state_memory      = np.load(destination + '/state_memory.npy')
        self.next_state_memory = np.load(destination + '/next_state_memory.npy')
        self.action_memory     = np.load(destination + '/action_memory.npy')
        self.reward_memory     = np.load(destination + '/reward_memory.npy')
        
    def sample_buffer(self, batch_size):
        
        max_mem = min(self.mem_cntr, self.mem_size)
        
        max_batch = min(max_mem, batch_size)
        
        batch = np.random.choice(max_mem, max_batch, replace=False)
                
        states            = self.state_memory[batch]
        rewards           = self.reward_memory[batch]
        actions           = self.action_memory[batch] 
        next_state_memory = self.next_state_memory[batch]
            
        return states, actions, rewards, next_state_memory
            


      

            
        
        

