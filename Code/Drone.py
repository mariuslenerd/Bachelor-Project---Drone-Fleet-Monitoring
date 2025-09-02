# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import os

class Drone():
    
    def __init__(self, x, y):
        
        #----- Coordinates -----
        
        self.x = x
        self.y = y
        
        self.list_of_move_actions = [0, 1, 2, 3, 4] #up, down, left, right, stay
        self.list_of_agent_types  = ['random']
        
        self.max_stay = 3
        self.stay_counter = 0
        self.at_peak = False
        
        #----- Store values for plotting -----
        
        self.list_of_observations     = []
        self.list_of_actions_taken    = [4]
        
        self.list_of_visited_nodes    = [[x,y]]
        self.history = []
    
    def random_init(self, Map):
        
        x_size = Map.x_size
        y_size = Map.y_size
        
        self.x = np.random.randint(0, high=x_size)
        self.y = np.random.randint(0, high=y_size)
        
    def info_adjacent_nodes(self, t, x, y, Map, mask, T):
        
        temporal_importance_values = []
        adjacent_nodes = [[x-1,y], [x+1,y], [x,y-1], [x,y+1], [x,y]] # same convention [up, down, left, right, stay]
        
        for given_x, given_y in adjacent_nodes:
            if 0 <= given_x <= Map.x_size-1 and 0 <= given_y <= Map.y_size-1:
                
                temporal_importance_values.append(Map.dynamic_importance(t, given_x, given_y, T ) * mask[given_x, given_y])
                
            else:
                
                temporal_importance_values.append(0.0)
                
        return adjacent_nodes, temporal_importance_values
    
    def get_observation(self, t, x, y, Map , mask, T, shared_info=None):
        
        observation = {}
        observation["x"] = x
        observation["y"] = y
        
        adjacent_nodes, temporal_importance_values = self.info_adjacent_nodes(t, x, y, Map, mask, T)
        
        observation["adjacent_nodes"] = adjacent_nodes
        observation["temporal_importance_values"] = temporal_importance_values 
        
        return observation
    
    def feasible(self, Map):
        
        given_x = self.x
        given_y = self.y

        feasible_actions = [4]
                
        if 0 <= given_x - 1:
            feasible_actions.append(0)
            
        if given_x + 1 <= Map.x_size - 1:
            feasible_actions.append(1)
            
        if 0 <= given_y - 1:
            feasible_actions.append(2)
            
        if given_y + 1 <= Map.y_size - 1:
            feasible_actions.append(3)
            
        return feasible_actions

#----- Different drone behaviours -----
    
    def random_agent_choose_action(self, observation):
        
        possible_actions = self.list_of_move_actions
        adjacent_temporal_importance_values = observation["temporal_importance_values"]
        
        feasible_actions = []
        for i in range(len(adjacent_temporal_importance_values)):
            
            ativ = adjacent_temporal_importance_values[i]
            action = possible_actions[i]
            
            if not ativ == 0.0:
                feasible_actions.append(action)
        
        if self.stay_counter == self.max_stay:
            
            self.stay_counter = 0
            feasible_actions = [action for action in feasible_actions if action!=4]
            
        move_action = random.choice(feasible_actions)
        if move_action == 4:
            self.stay_counter += 1
        
        return move_action
      
    def SLS_agent_choose_action(self, observation):
        
        move_action = None  # Initialize move_action to None
        
        importance_values = observation["temporal_importance_values"]

        if not self.at_peak:
            # Find the index of the maximum importance value
            max_importance_index = np.argmax(importance_values)

            # Check if the maximum importance value is zero or negative
            if importance_values[max_importance_index] <= 0:
                # If all importance values are zero or negative, choose a random action
                move_action = np.random.choice(self.list_of_move_actions)
            else:
                # Otherwise, choose the action corresponding to the maximum importance value
                move_action = self.list_of_move_actions[max_importance_index]

                # Check if the drone has reached the peak
                if max_importance_index == 4:  # Index of "stay" action
                    self.at_peak = True
        else:
            # Implement patrolling behavior
            # Choose the action based on patrolling strategy (e.g., moving to adjacent nodes)
            # For simplicity, let's implement a simple patrolling strategy of moving to the next adjacent node cyclically
            current_index = self.list_of_move_actions.index(move_action)
            next_index = (current_index + 1) % len(self.list_of_move_actions)
            move_action = self.list_of_move_actions[next_index]

        return move_action

    def greedy_agent_choose_action(self, observation):
        
        possible_actions = self.list_of_move_actions
        adjacent_temporal_importance_values = observation["temporal_importance_values"]
        
        feasible_actions = []
        feasible_action_values = []
        
        for i in range(len(adjacent_temporal_importance_values)):
            
            ativ = adjacent_temporal_importance_values[i]
            action = possible_actions[i]
            
            if not ativ == 0:
                feasible_actions.append(action)
                feasible_action_values.append(ativ)
        
        if self.stay_counter == self.max_stay:
            
            self.stay_counter = 0
            
            feasible_actions = [action for action in feasible_actions if action!=4]
            feasible_action_values.pop()
            
        max_index=feasible_action_values.index(max(feasible_action_values))    
        move_action=feasible_actions[max_index]
    
        if move_action == 4:
            self.stay_counter += 1
        
        return move_action
       
    def round_agent(self, observation):
        
        possible_actions = self.list_of_move_actions
        
        move_action = None  # Initialize move_action
        
        if self.x == 5 and self.y < 25:
            move_action = possible_actions[3]
        elif self.y == 25 and self.x < 15:
            move_action = possible_actions[1]
        elif self.x == 15 and self.y > 5:
            move_action = possible_actions[2]
        elif self.y == 5 and self.x > 5:
            move_action = possible_actions[0]
        elif self.x == 5 and self.y == 5:  # Starting point, move right to initiate the loop
            move_action = possible_actions[3]
        
        return move_action

    def explorer_agent(self, observation):
        
        possible_actions = self.list_of_move_actions
        
        move_action = None  # Initialize move_action
        
        if self.y % 4 == 0 and self.x!=19 and self.list_of_actions_taken[-1]!=2:
            move_action = possible_actions[1]
            self.list_of_actions_taken.append(move_action)
        elif (self.y % 4 == 0 or self.y % 4 == 1) and self.x==19 and self.y!=29:
            move_action = possible_actions[3]
            self.list_of_actions_taken.append(move_action)
        elif self.y % 4 == 2 and self.x!=0:
            move_action = possible_actions[0]
            self.list_of_actions_taken.append(move_action)
        elif (self.y % 4 == 2 or self.y % 4==3) and self.x==0 and self.list_of_actions_taken[-1]!=2:
            move_action = possible_actions[3]
            self.list_of_actions_taken.append(move_action)
        elif self.y == 29 and self.x!=0:
            move_action = possible_actions[0]
            self.list_of_actions_taken.append(move_action)
        elif self.y == 29 and self.x==0:
            move_action = possible_actions[2]
            self.list_of_actions_taken.append(move_action)
        elif self.x==0 and self.y!=0 and self.list_of_actions_taken[-1]==2 :
            move_action = possible_actions[2]
            self.list_of_actions_taken.append(move_action)
  
        return move_action
            
    def choose_action(self, observation, agent_type='random'):
        
        if agent_type == 'random':
            
            move_action = self.random_agent_choose_action(observation)
            
        elif agent_type == 'G'   :
            
            move_action = self.greedy_agent_choose_action(observation)
        
        elif agent_type == 'SLS':
            
            move_action = self.SLS_agent_choose_action(observation)
        
        elif agent_type == 'round':
            
            move_action = self.round_agent(observation)

        elif agent_type == 'exp':
            
            move_action = self.explorer_agent(observation)         
            
        return move_action
    
    def update_drone(self, observation, move_action):
        
        if move_action == - np.infty:
            
            self.x = self.x
            self.y = self.y
            
        elif move_action == 0:
            self.x -= 1
        elif move_action == 1:
            self.x += 1 
        elif move_action == 2:
            self.y -= 1
        elif move_action == 3:
            self.y += 1
        
        self.list_of_observations.append(observation)
        self.list_of_actions_taken.append(move_action)
        self.list_of_visited_nodes.append([self.x, self.y])
        self.history.append((self.x,self.y))