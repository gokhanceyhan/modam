#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:07:42 2018

@author: gokhanceyhan
"""

import pandas as pd

class DamData:
    
    # minimum allowable bid price
    min_price = 0
    
    # maximum allowable bid price
    max_price = 2000
    
    # number of time periods in dam
    number_of_periods = 24
    
    # reading the dam input from the file specified
    def read_input(self, file_path):
        data = pd.read_csv(file_path)
        
    
    # printing stats for the given dam data
    def print_input_stats(self):
        pass
    
class DamBids:
    
    def __init__(self, hourly_bids, block_bids, flexible_bids):
        self.hourly_bids = hourly_bids
        self.block_bids = block_bids
        self.flexible_bids = flexible_bids
        
    
class Bid:
    
    def __init__(self, id, num_period):
        self.id = id
        self.num_period = num_period
    

class HourlyBid(Bid):
    
    def __init__(self, id, num_period, period, price_quantity_pairs):
        Bid.__init__(self, id, num_period)
        self.period = period
        self.price_quantity_pairs = price_quantity_pairs
        
        
class BlockBid(Bid):
    
    def __init__(self, id, num_period, period, price, quantity):
        Bid.__init__(self, id, num_period)
        self.period = period
        self.price = price
        self.quantity = quantity


class FlexibleBid(Bid):

    def __init__(self, id, num_period, price, quantity):
        Bid.__init__(self, id, num_period)
        self.price = price
        self.quantity = quantity
        
    
        
       
        