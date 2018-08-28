#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 21:07:42 2018

@author: gokhanceyhan
"""

import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from dam_constants import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DamData:
    
    # minimum allowable bid price
    min_price = 0
    
    # maximum allowable bid price
    max_price = 2000
    
    # number of time periods in dam
    number_of_periods = 24
    
    # reading the dam input from the file specified
    def read_input(self, file_path):
        '''
        Reads input data in the specified path and create dam_bids object
        
        :param file_path: relative path of the input file
        '''
        
        # read data as a data frame
        data = pd.read_csv(file_path)
        
        # define bid maps
        bidId_to_hourlyBid = {}
        bidId_to_blockBid = {}
        bidId_to_flexibleBid = {}

        # iterate data frame and create bid objects
        for index, row in data.iterrows():
            
            type = row[DAM_DATA_BID_TYPE_HEADER]
            id = row[DAM_DATA_BID_ID_HEADER]
            num_of_periods = row[DAM_DATA_BID_NUMBER_OF_PERIODS_HEADER]
            period = row[DAM_DATA_BID_PERIOD_HEADER]
            price = row[DAM_DATA_BID_PRICE_HEADER]
            quantity = row[DAM_DATA_BID_QUANTITY_HEADER]
            link = row[DAM_DATA_BID_LINK_HEADER]
            
            if BidType(type) is BidType.BLOCK:
                block_bid = BlockBid(id, num_of_periods, period, price,
                                     quantity, link )
                bidId_to_blockBid.update({block_bid.id : block_bid})
                
            elif BidType(type) is BidType.FLEXIBLE:
                flexible_bid = FlexibleBid(id, num_of_periods, price, quantity)
                bidId_to_flexibleBid.update({flexible_bid.id : flexible_bid })
                
            elif id not in bidId_to_hourlyBid:
                hourly_bid = HourlyBid(id, num_of_periods, period, price, 
                                       quantity)
                bidId_to_hourlyBid.update({hourly_bid.id : hourly_bid})
                
            else:
                bidId_to_hourlyBid[id].add_price_quantity_pair(price, quantity)
                
        self.dam_bids = DamBids(bidId_to_hourlyBid, bidId_to_blockBid, 
                                bidId_to_flexibleBid)
        
    # printing stats for the given dam data

    
class DamBids:
    
    def __init__(self, hourly_bids, block_bids, flexible_bids):
        self.hourly_bids = hourly_bids
        self.block_bids = block_bids
        self.flexible_bids = flexible_bids
        
    
class Bid(ABC):
    
    def __init__(self, id, type, num_period):
        self.id = id
        self.type = type
        self.num_period = num_period
        super().__init__()
        
    @abstractmethod
    def print_bid(self):
        pass
    

class HourlyBid(Bid):
    
    def __init__(self, id, num_period, period, price, quantity):
        super().__init__(id, BidType.HOURLY, num_period)
        self.period = period
        self.price_quantity_pairs = [(price, quantity)]
        
    def add_price_quantity_pair(self, price, quantity):
        '''
        Adds a price-quantity pair to the current list
        
        :param price: price value
        :param quantity: quantity value
        '''
        self.price_quantity_pairs.append((price, quantity))
        
    def print_bid(self):
        pass
        
        
class BlockBid(Bid):
    
    def __init__(self, id, num_period, period, price, quantity, link):
        super().__init__(id, BidType.BLOCK, num_period)
        self.period = period
        self.price = price
        self.quantity = quantity
        self.link = link
        
    def print_bid(self):
        pass


class FlexibleBid(Bid):

    def __init__(self, id, num_period, price, quantity):
        super().__init__(id, BidType.FLEXIBLE, num_period)
        self.price = price
        self.quantity = quantity
        
    def print_bid(self):
        pass
    
        
class BidType(Enum):
    HOURLY = 'S'
    BLOCK = 'B'
    FLEXIBLE = 'F'   
    
class InputStats():
    
    def __init__(self, dam_data):
        self.number_of_hourlyBids = len(dam_data.dam_bids.hourly_bids)
        self.number_of_blockBids = len(dam_data.dam_bids.block_bids)
        self.number_of_flexibleBids = len(dam_data.dam_bids.flexible_bids)
        
    def print_stats(self):
        logger.info('Printing input stats...')
        logger.info('Number of hourly bids: %s', self.number_of_hourlyBids)
        logger.info('Number of block bids: %s', self.number_of_blockBids)
        logger.info('Number of flexible bids: %s', self.number_of_flexibleBids)
        