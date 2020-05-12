"""
Created on Thu Aug  2 21:07:42 2018

@author: gokhanceyhan
"""

from abc import abstractmethod
from enum import Enum
import logging
import pandas as pd

import modam.surplus_maximization.dam_constants as dc
import modam.surplus_maximization.dam_utils as du

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

            bid_type = row[dc.DAM_DATA_BID_TYPE_HEADER]
            bid_id = str(row[dc.DAM_DATA_BID_ID_HEADER])
            num_of_periods = row[dc.DAM_DATA_BID_NUMBER_OF_PERIODS_HEADER]
            period = row[dc.DAM_DATA_BID_PERIOD_HEADER]
            price = row[dc.DAM_DATA_BID_PRICE_HEADER]
            quantity = row[dc.DAM_DATA_BID_QUANTITY_HEADER]
            link = row[dc.DAM_DATA_BID_LINK_HEADER]

            if bid_type == BidType.BLOCK.value:
                block_bid = BlockBid(bid_id, num_of_periods, period, price, quantity, link)
                bid_id_2_block_bid.update({block_bid.bid_id: block_bid})
            elif bid_type == BidType.FLEXIBLE.value:
                flexible_bid = FlexibleBid(bid_id, num_of_periods, price, quantity)
                bid_id_2_flexible_bid.update({flexible_bid.bid_id: flexible_bid})
            elif bid_id not in bid_id_2_hourly_bid:
                hourly_bid = HourlyBid(bid_id, num_of_periods, period, price, quantity)
                bid_id_2_hourly_bid.update({hourly_bid.bid_id: hourly_bid})
            else:
                bid_id_2_hourly_bid[bid_id].add_price_quantity_pair(price, quantity)
        # create simple bids from hourly bids
        for bid_id, hourly_bid in bid_id_2_hourly_bid.items():
            hourly_bid.step_id_2_simple_bid = du.create_simple_bids_from_hourly_bid(hourly_bid)
        self.dam_bids = DamBids(bid_id_2_hourly_bid, bid_id_2_block_bid, bid_id_2_flexible_bid)


class DamBids:

    def __init__(self, bid_id_2_hourly_bid, bid_id_2_block_bid, bid_id_2_flexible_bid):
        self.bid_id_2_hourly_bid = bid_id_2_hourly_bid
        self.bid_id_2_block_bid = bid_id_2_block_bid
        self.bid_id_2_flexible_bid = bid_id_2_flexible_bid


class Bid(object):

    def __init__(self, bid_id, bid_type, num_period):
        self.bid_id = bid_id
        self.type = bid_type
        self.num_period = num_period

    @abstractmethod
    def print_bid(self):
        pass


class HourlyBid(Bid):

    def __init__(self, bid_id, num_period, period, price, quantity):
        """
        Hourly bid class

        :param bid_id:
        :param num_period:
        :param period:
        :param price:
        :param quantity:
        """
        Bid.__init__(self, bid_id, BidType.HOURLY, num_period)
        self.period = period
        self.price_quantity_pairs = [(price, quantity)]
        self.step_id_2_simple_bid = None

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
        Bid.__init__(self, bid_id, BidType.BLOCK, num_period)
        self.period = period
        self.price = price
        self.quantity = quantity
        self.link = link
        if quantity <= 0:
            self.is_supply = True
        else:
            self.is_supply = False

    def print_bid(self):
        pass


class FlexibleBid(Bid):

    def __init__(self, bid_id, num_period, price, quantity):
        Bid.__init__(self, bid_id, BidType.FLEXIBLE, num_period)
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
        self._number_of_hourly_bids = len(dam_data.dam_bids.bid_id_2_hourly_bid)
        self._number_of_block_bids = len(dam_data.dam_bids.bid_id_2_block_bid)
        self._number_of_flexible_bids = len(dam_data.dam_bids.bid_id_2_flexible_bid)

    def number_of_hourly_bids(self):
        """Returns the number of hourly bids"""
        return self._number_of_hourly_bids

    def number_of_block_bids(self):
        """Returns the number of block bids"""
        return self._number_of_block_bids

    def number_of_flexible_bids(self):
        """Returns the number of flexible bids"""
        return self._number_of_flexible_bids

    def print_stats(self):
        logger.info('Printing input stats...')
        logger.info('Number of hourly bids: %s', self._number_of_hourly_bids)
        logger.info('Number of block bids: %s', self._number_of_block_bids)
        logger.info('Number of flexible bids: %s', self._number_of_flexible_bids)
