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

    def __init__(self):
        self.dam_bids = None

    # minimum allowable bid price
    min_price = 0

    # maximum allowable bid price
    max_price = 2000

    # number of time periods in dam
    number_of_periods = 24

    # reading the dam input from the file specified
    def read_input(self, file_path):
        """
        Reads input data in the specified path and create dam_bids object
        :param file_path: relative path of the input file
        :return:
        """
        # read data as a data frame
        data = pd.read_csv(file_path)

        # define bid maps
        bid_id_2_hourly_bid = {}
        bid_id_2_block_bid = {}
        bid_id_2_flexible_bid = {}

        # iterate data frame and create bid objects
        for index, row in data.iterrows():

            bid_type = row[DAM_DATA_BID_TYPE_HEADER]
            bid_id = row[DAM_DATA_BID_ID_HEADER]
            num_of_periods = row[DAM_DATA_BID_NUMBER_OF_PERIODS_HEADER]
            period = row[DAM_DATA_BID_PERIOD_HEADER]
            price = row[DAM_DATA_BID_PRICE_HEADER]
            quantity = row[DAM_DATA_BID_QUANTITY_HEADER]
            link = row[DAM_DATA_BID_LINK_HEADER]

            if BidType(bid_type) is BidType.BLOCK:
                block_bid = BlockBid(bid_id, num_of_periods, period, price,
                                     quantity, link)
                bid_id_2_block_bid.update({block_bid.bid_id: block_bid})

            elif BidType(bid_type) is BidType.FLEXIBLE:
                flexible_bid = FlexibleBid(bid_id, num_of_periods, price, quantity)
                bid_id_2_flexible_bid.update({flexible_bid.bid_id: flexible_bid})

            elif bid_id not in bid_id_2_hourly_bid:
                hourly_bid = HourlyBid(bid_id, num_of_periods, period, price,
                                       quantity)
                bid_id_2_hourly_bid.update({hourly_bid.bid_id: hourly_bid})

            else:
                bid_id_2_hourly_bid[bid_id].add_price_quantity_pair(price, quantity)

        self.dam_bids = DamBids(bid_id_2_hourly_bid, bid_id_2_block_bid,
                                bid_id_2_flexible_bid)

    # printing stats for the given dam data


class DamBids:

    def __init__(self, hourly_bids, block_bids, flexible_bids):
        self.hourly_bids = hourly_bids
        self.block_bids = block_bids
        self.flexible_bids = flexible_bids


class Bid(ABC):

    def __init__(self, bid_id, bid_type, num_period):
        self.bid_id = bid_id
        self.type = bid_type
        self.num_period = num_period
        super().__init__()

    @abstractmethod
    def print_bid(self):
        pass


class HourlyBid(Bid):

    def __init__(self, bid_id, num_period, period, price, quantity):
        super().__init__(bid_id, BidType.HOURLY, num_period)
        self.period = period
        self.price_quantity_pairs = [(price, quantity)]

    def add_price_quantity_pair(self, price, quantity):
        """
        Adds a price-quantity pair to the current list
        :param price:
        :param quantity:
        :return:
        """
        self.price_quantity_pairs.append((price, quantity))

    def print_bid(self):
        pass


class BlockBid(Bid):

    def __init__(self, bid_id, num_period, period, price, quantity, link):
        super().__init__(bid_id, BidType.BLOCK, num_period)
        self.period = period
        self.price = price
        self.quantity = quantity
        self.link = link

    def print_bid(self):
        pass


class FlexibleBid(Bid):

    def __init__(self, bid_id, num_period, price, quantity):
        super().__init__(bid_id, BidType.FLEXIBLE, num_period)
        self.price = price
        self.quantity = quantity

    def print_bid(self):
        pass


class BidType(Enum):
    HOURLY = 'S'
    BLOCK = 'B'
    FLEXIBLE = 'F'


class InputStats:

    def __init__(self, dam_data):
        self.number_of_hourlyBids = len(dam_data.dam_bids.hourly_bids)
        self.number_of_blockBids = len(dam_data.dam_bids.block_bids)
        self.number_of_flexibleBids = len(dam_data.dam_bids.flexible_bids)

    def print_stats(self):
        logger.info('Printing input stats...')
        logger.info('Number of hourly bids: %s', self.number_of_hourlyBids)
        logger.info('Number of block bids: %s', self.number_of_blockBids)
        logger.info('Number of flexible bids: %s', self.number_of_flexibleBids)
