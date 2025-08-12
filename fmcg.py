import os
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request
import plotly.graph_objects as go
import threading
import time

class InventoryEnv(gym.Env):
    metadata = {'render_modes': ['human']} #included for good practice - actually not used in project 

    def __init__(self, config=None, filepath='train.csv', store_id=1, item_id=1):
        super(InventoryEnv, self).__init__()

        # <<< CHANGE START: LOAD AND PROCESS REAL DATA >>>
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            # Filter for a specific store and item
            self.demand_data = df[(df['store'] == store_id) & (df['item'] == item_id)]['sales'].values
            if len(self.demand_data) == 0:
                raise ValueError(f"No data found for store {store_id}, item {item_id}. Please check IDs.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: '{filepath}' not found. Please download it and place it in the same directory.")
        # <<< CHANGE END >>>

        default_config = {
            'max_inventory': 150,
            'product_cost': 2,
            'holding_cost': 0.1,
            'stockout_penalty': 10,
            'selling_price': 5,
            # <<< CHANGE: Episode length is now determined by the data length >>>
            'episode_length': len(self.demand_data),
        }
        if config:
            default_config.update(config)

        self.config = default_config
        self.max_inventory = self.config['max_inventory']
        
        self.action_space = spaces.Box(low=0, high=self.max_inventory, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.max_inventory, shape=(1,), dtype=np.float32)
        
        self.current_step = 0
        self.inventory = 0

    def _get_obs(self):
        return np.array([self.inventory], dtype=np.float32)

    def _get_info(self):
        return {"inventory": self.inventory}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = self.np_random.uniform(low=0, high=20)
        self.current_step = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, 0, self.max_inventory - self.inventory)[0]
        
        # <<< CHANGE START: GET DEMAND FROM DATASET INSTEAD OF SIMULATING >>>
        if self.current_step < len(self.demand_data):
            demand = self.demand_data[self.current_step]
        else:
            demand = 0 # End of data
        # <<< CHANGE END >>>
        
        order_cost = action * self.config['product_cost']
        self.inventory += action
        
        sales = min(self.inventory, demand)
        lost_sales = demand - sales
        
        revenue = sales * self.config['selling_price']
        stockout_cost = lost_sales * self.config['stockout_penalty']
        
        self.inventory -= sales
        holding_cost = self.inventory * self.config['holding_cost']
        
        reward = revenue - order_cost - holding_cost - stockout_cost
        
        self.current_step += 1
        terminated = self.current_step >= self.config['episode_length']
        
        return self._get_obs(), reward, terminated, False, {"daily_profit": reward}

