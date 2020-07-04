"""
Created on Thu Aug  2 21:07:42 2018

@author: gokhanceyhan
"""

from abc import abstractmethod
from enum import Enum
import logging
import numpy as np
import pandas as pd

import modam.surplus_maximization.dam_constants as dc
import modam.surplus_maximization.dam_utils as du

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DamData:

    _FIELD_NAME_2_DATA_TYPE = {
        "bid_id": str,
        "bucket_id": "int16",
        "period": "int16",
        "bid_type": str,
        "zone": str,
        "quantity": "float64",
        "price": "float64",
        "num_periods": "int16",
        "link": str
    }

    # minimum allowable bid price
    min_price = 0
    # maximum allowable bid price
    max_price = 2000
    # number of time periods in dam
    number_of_periods = 24
    # reading the dam input from the file specified

    def __init__(self):
        self.dam_bids = None

    @staticmethod
    def _create_idential_bid_lists(bids_df):
        """Create the list of identical bid sets for different bid types
        
        NOTE: Currently, only block bids are checked."""
        bid_type_2_identical_bid_lists = {}
        block_bids_df = bids_df[(bids_df["bid_type"] == BidType.BLOCK.value) & (bids_df["link"].isnull())].reset_index(
            drop=True)
        bid_lists = []
        for name, df_ in block_bids_df.groupby(by=["period", "num_periods", "quantity", "price"]):
            if df_["bid_id"].count() == 1:
                continue
            sorted_df = df_.sort_values("bid_id")
            identical_bids = []
            for _, row in sorted_df.iterrows():
                bid_type = row[dc.DAM_DATA_BID_TYPE_HEADER]
                bid_id = row[dc.DAM_DATA_BID_ID_HEADER]
                num_of_periods = row[dc.DAM_DATA_BID_NUMBER_OF_PERIODS_HEADER]
                period = row[dc.DAM_DATA_BID_PERIOD_HEADER]
                price = row[dc.DAM_DATA_BID_PRICE_HEADER]
                quantity = row[dc.DAM_DATA_BID_QUANTITY_HEADER]
                link = row[dc.DAM_DATA_BID_LINK_HEADER]
                block_bid = BlockBid(bid_id, num_of_periods, period, price, quantity, link)
                identical_bids.append(block_bid)
            bid_lists.append(identical_bids)
        bid_type_2_identical_bid_lists[BidType.BLOCK] = bid_lists
        return bid_type_2_identical_bid_lists

    def read_input(self, file_path):
        """
        Reads input data in the specified path and create dam_bids object
        :param file_path: relative path of the input file
        :return:
        """
        # read data as a data frame
        df = pd.read_csv(file_path, dtype=DamData._FIELD_NAME_2_DATA_TYPE)
        df = df.iloc[:, 4:]
        # define bid maps
        bid_id_2_hourly_bid = {}
        bid_id_2_block_bid = {}
        bid_id_2_flexible_bid = {}
        # iterate data frame and create bid objects
        for index, row in df.iterrows():

            bid_type = row[dc.DAM_DATA_BID_TYPE_HEADER]
            bid_id = row[dc.DAM_DATA_BID_ID_HEADER]
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
        # create the identical bid lists
        bid_type_2_identical_bid_lists = DamData._create_idential_bid_lists(df)
        self.dam_bids = DamBids(
            bid_id_2_hourly_bid, bid_id_2_block_bid, bid_id_2_flexible_bid, 
            bid_type_2_identical_bid_lists=bid_type_2_identical_bid_lists)


class DamBids:

    def __init__(
            self, bid_id_2_hourly_bid, bid_id_2_block_bid, bid_id_2_flexible_bid, 
            bid_type_2_identical_bid_lists=None):
        self.bid_id_2_hourly_bid = bid_id_2_hourly_bid
        self.bid_id_2_block_bid = bid_id_2_block_bid
        self.bid_id_2_flexible_bid = bid_id_2_flexible_bid
        self.bid_type_2_identical_bid_lists = bid_type_2_identical_bid_lists or {}


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
        self._average_block_bid_num_period = np.mean(
            [block_bid.num_period for block_bid in dam_data.dam_bids.bid_id_2_block_bid.values()])
        self._average_block_bid_quantity = np.mean(
            [abs(block_bid.quantity) for block_bid in dam_data.dam_bids.bid_id_2_block_bid.values()])
        self._number_of_hourly_bids = len(dam_data.dam_bids.bid_id_2_hourly_bid)
        self._number_of_block_bids = len(dam_data.dam_bids.bid_id_2_block_bid)
        self._number_of_flexible_bids = len(dam_data.dam_bids.bid_id_2_flexible_bid)
    
    def average_block_bid_num_period(self):
        """Returns the average block bid number of periods"""
        return self._average_block_bid_num_period

    def average_block_bid_quantity(self):
        """Returns the average block bid quantity"""
        return self._average_block_bid_quantity

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
