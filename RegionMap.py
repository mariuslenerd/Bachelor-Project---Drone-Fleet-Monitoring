# -*- coding: utf-8 -*-
import numpy as np
import os 
from datetime import datetime
import matplotlib.pyplot as plt

class RegionMap():
    
    def __init__(self, v_size, h_size):
        
       self.y_size = h_size
       self.x_size = v_size
       
       self.importance_map_s = np.zeros((self.x_size, self.y_size))
       self.importance_map_b = np.zeros((self.x_size, self.y_size))
       
       self.importance_map = np.zeros((self.x_size, self.y_size))
    
    def initialize_importance_map(self, list_of_small_pertb, list_of_big_pert):
        
        small_amplitude = 0.5
        small_sigma     = 1.0
        
        big_amplitude   = 1.0
        big_sigma       = 3.0
        
        for x in range(self.x_size):
            for y in range(self.y_size):        
                
                for small_p in list_of_small_pertb:
                    x_p = small_p[0]
                    y_p = small_p[1] 
                    self.importance_map_s[x,y] = self.importance_map_s[x,y] + small_amplitude * np.exp(-((x-x_p)**2+(y-y_p)**2)/small_sigma**2/2.0)
                
                for big_p in list_of_big_pert:
                    x_p = big_p[0]
                    y_p = big_p[1]      
                    self.importance_map_b[x,y] = self.importance_map_b[x,y] + big_amplitude   * np.exp(-((x-x_p)**2+(y-y_p)**2)/big_sigma**  2/2.0)
                
                self.importance_map[x, y] = self.importance_map_b[x, y] + self.importance_map_s[x, y]
    
    def write_to_file(self, filename=None):
        
        path = os.getcwd() + '/Map_configurations'
        if not os.path.isdir(path):
            os.makedirs(path)
        
        if filename == None:
            
            now = datetime.now()
            date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
            filename = + date_time + '_' + 'map.npy'
            
        else:
            
            filename += '_map.npy'
        
        name = path + '/' + filename                       
        
        np.save(name, self.importance_map)

    def load_from_file(self, filename):

        path = os.getcwd() + '/Map_configurations'
        file_path = path + '/' + filename
        
        data = np.load(file_path)
        imp_map = np.array(data, dtype=np.float32)

        self.v_size = np.shape(imp_map)[0]
        self.h_size = np.shape(imp_map)[1]
        
        self.importance_map = imp_map

    def plot_map(self, rmap_values=True):

        
        fig, ax = plt.subplots()
        _ = plt.imshow(self.importance_map, cmap = 'viridis')

        # ----- Annotate values on the RegionMap -----
        
        if rmap_values:
            for i in range(self.x_size):
                for j in range(self.y_size):
                    _ = ax.text(i, j, f'{self.importance_map[i, j]:.2f}', ha='center', va='center', color='black')


        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")

        cbar = plt.colorbar()
        cbar.ax.set_ylabel("", rotation=90)
     
    def plot_drones_on_map(self, list_of_drone_positions):

        fig, ax = plt.subplots()
        _ = plt.imshow(self.importance_map, cmap = 'viridis')

        plt.xlabel("y-coordinate")
        plt.ylabel("x-coordinate")

        cbar = plt.colorbar()
        cbar.ax.set_ylabel("", rotation=90)

        list_of_markers = ["o", "v", "s", "p", "D"]
        
        for idx, dpos in enumerate(list_of_drone_positions):
            
            marker_idx = list_of_markers[idx]
            plt.scatter(dpos[1], dpos[0], color='white', facecolors='none',linewidth=1,marker=marker_idx,s=200)
            
    def dynamic_importance(self, t, x, y, T):
              
        importance_map_s_2 = np.zeros((self.x_size, self.y_size))
        importance_map_b_2 = np.zeros((self.x_size, self.y_size))        
        
        if np.sin(2 * np.pi * (5/T) * t) > 0:
            
            importance_map_s_2[x, y]=np.sin(2 * np.pi * (5/T) * t)* self.importance_map_s[x, y]
            
        else:
            importance_map_s_2[x, y]=0
                
        
        importance_map_b_2[x, y] = (np.exp(-t/(0.7*T))) * self.importance_map_b[x, y]
        
        self.importance_map[x, y] = importance_map_s_2[x, y] + importance_map_b_2[x, y]
    
        
        return self.importance_map[x,y]

