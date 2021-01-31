"""
Created on Thu Aug  2 21:07:42 2018

@author: gokhanceyhan
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
import logging
import numpy as np
import pandas as pd

import modam.surplus_maximization.dam_constants as dc
from modam.surplus_maximization.dam_utils import interpolate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PriceQuantityPair = namedtuple("PriceQuantityPair", ["p", "q"])
SimpleBid = namedtuple("SimpleBid", ["p", "q"])
InterpolatedBid = namedtuple("InterpolatedBid", ["p_start", "p_end", "q"])


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
    MIN_PRICE = 0
    # maximum allowable bid price
    MAX_PRICE = 2000
    # number of time periods in dam
    NUM_PERIODS = 24
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

    @staticmethod
    def _create_interpolated_bids_from_piecewise_hourly_bid(hourly_bid):
        """Creates interpolated bids (price-quantity-pair tuple) that form the given piecewise hourly bid"""
        step_id_2_interpolated_bid = {}
        count = 1
        prev_qnt = 0
        if hourly_bid.price_quantity_pairs[0].q <= 0:
            # supply bid
            prev_prc = DamData.MIN_PRICE
            for price, quantity in hourly_bid.price_quantity_pairs:
                step_id_2_interpolated_bid[count] = InterpolatedBid(
                    p_start=prev_prc, p_end=price, q=quantity - prev_qnt)
                prev_qnt = quantity
                prev_prc = price
                count += 1
        elif hourly_bid.price_quantity_pairs[-1].q >= 0:
            # demand bid
            prev_prc = DamData.MAX_PRICE
            for price, quantity in reversed(hourly_bid.price_quantity_pairs):
                step_id_2_interpolated_bid[count] = InterpolatedBid(
                    p_start=prev_prc, p_end=price, q=quantity - prev_qnt)
                prev_qnt = quantity
                prev_prc = price
                count += 1
        else:
            prev_qnt = hourly_bid.price_quantity_pairs[0].q
            prev_prc = hourly_bid.price_quantity_pairs[0].p
            for price, quantity in hourly_bid.price_quantity_pairs[1:]:
                if quantity >= 0:
                    # demand step
                    step_id_2_interpolated_bid[count] = InterpolatedBid(
                        p_start=price, p_end=prev_prc, q=prev_qnt - quantity)
                elif prev_qnt > 0:
                    # first supply step: we need to divide the bid into supply and demand bids
                    zero_quantity_price = interpolate(prev_prc, prev_qnt, price, quantity, q=0)
                    step_id_2_interpolated_bid[count] = InterpolatedBid(
                        p_start=zero_quantity_price, p_end=prev_prc, q=prev_qnt)
                    count += 1
                    step_id_2_interpolated_bid[count] = InterpolatedBid(
                        p_start=zero_quantity_price, p_end=price, q=quantity)
                else:
                    # supply step
                    step_id_2_interpolated_bid[count] = InterpolatedBid(
                        p_start=prev_prc, p_end=price, q=quantity - prev_qnt)
                prev_qnt = quantity
                prev_prc = price
                count += 1
        return {
            step_id: interpolated_bid for step_id, interpolated_bid in step_id_2_interpolated_bid.items() if 
            abs(interpolated_bid.q) > 0}

    @staticmethod
    def _create_simple_bids_from_step_hourly_bid(hourly_bid):
        """Creates simple bids (price-quantity pairs) that form the given step hourly bid"""
        step_id_2_simple_bid = {}
        count = 1
        prev_qnt = 0
        if hourly_bid.price_quantity_pairs[0].q <= 0:
            # supply bid
            for price, quantity in hourly_bid.price_quantity_pairs:
                step_id_2_simple_bid[count] = SimpleBid(p=price, q=quantity - prev_qnt)
                prev_qnt = quantity
                count += 1
        elif hourly_bid.price_quantity_pairs[-1].q >= 0:
            # demand bid
            for price, quantity in reversed(hourly_bid.price_quantity_pairs):
                step_id_2_simple_bid[count] = SimpleBid(p=price, q=quantity - prev_qnt)
                prev_qnt = quantity
                count += 1
        else:
            prev_qnt = hourly_bid.price_quantity_pairs[0].q
            prev_prc = hourly_bid.price_quantity_pairs[0].p
            for price, quantity in hourly_bid.price_quantity_pairs[1:]:
                if quantity >= 0:
                    # demand step
                    step_id_2_simple_bid[count] = SimpleBid(p=prev_prc, q=prev_qnt - quantity)
                    prev_prc = price
                elif prev_qnt > 0:
                    # first supply step
                    step_id_2_simple_bid[count] = SimpleBid(p=prev_prc, q=prev_qnt)
                    count += 1
                    step_id_2_simple_bid[count] = SimpleBid(p=price, q=quantity)
                else:
                    # supply step
                    step_id_2_simple_bid[count] = SimpleBid(p=price, q=quantity - prev_qnt)
                prev_qnt = quantity
                count += 1
        return {step_id: simple_bid for step_id, simple_bid in step_id_2_simple_bid.items() if abs(simple_bid.q) > 0}

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
        bid_id_2_step_hourly_bid = {}
        bid_id_2_piecewise_hourly_bid = {}
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
            elif bid_type == BidType.STEP_HOURLY.value:
                if bid_id not in bid_id_2_step_hourly_bid:
                    hourly_bid = StepHourlyBid(bid_id, period)
                    hourly_bid.add_price_quantity_pair(price, quantity)
                    bid_id_2_step_hourly_bid[bid_id] = hourly_bid
                else:
                    bid_id_2_step_hourly_bid[bid_id].add_price_quantity_pair(price, quantity)
            elif bid_type == BidType.PIECEWISE_HOURLY.value:
                if bid_id not in bid_id_2_piecewise_hourly_bid:
                    hourly_bid = PiecewiseHourlyBid(bid_id, period)
                    hourly_bid.add_price_quantity_pair(price, quantity)
                    bid_id_2_piecewise_hourly_bid[bid_id] = hourly_bid
                else:
                    bid_id_2_piecewise_hourly_bid[bid_id].add_price_quantity_pair(price, quantity)
        # create simple bids from step hourly bids
        for bid_id, hourly_bid in bid_id_2_step_hourly_bid.items():
            hourly_bid.step_id_2_simple_bid = DamData._create_simple_bids_from_step_hourly_bid(hourly_bid)
        # create simple bids from piecewise hourly bids
        for bid_id, hourly_bid in bid_id_2_piecewise_hourly_bid.items():
            hourly_bid.step_id_2_interpolated_bid = \
                DamData._create_interpolated_bids_from_piecewise_hourly_bid(hourly_bid)
        # create the identical bid lists
        bid_type_2_identical_bid_lists = DamData._create_idential_bid_lists(df)
        self.dam_bids = DamBids(
            bid_id_2_step_hourly_bid, bid_id_2_piecewise_hourly_bid, bid_id_2_block_bid, bid_id_2_flexible_bid, 
            bid_type_2_identical_bid_lists=bid_type_2_identical_bid_lists)


class DamBids:

    def __init__(
            self, bid_id_2_step_hourly_bid, bid_id_2_piecewise_hourly_bid, bid_id_2_block_bid, bid_id_2_flexible_bid, 
            bid_type_2_identical_bid_lists=None):
        self.bid_id_2_step_hourly_bid = bid_id_2_step_hourly_bid
        self.bid_id_2_piecewise_hourly_bid = bid_id_2_piecewise_hourly_bid
        self.bid_id_2_block_bid = bid_id_2_block_bid
        self.bid_id_2_flexible_bid = bid_id_2_flexible_bid
        self.bid_type_2_identical_bid_lists = bid_type_2_identical_bid_lists or {}


class Bid(ABC):

    def __init__(self, bid_id, num_period):
        self.bid_id = bid_id
        self.num_period = num_period

    @abstractmethod
    def print_bid(self):
        pass


class PiecewiseHourlyBid(Bid):

    """Implements a piece-wise (linear) hourly bid"""

    def __init__(self, bid_id, period):
        num_period = 1
        super(PiecewiseHourlyBid, self).__init__(bid_id, num_period)
        self.period = period
        self.price_quantity_pairs = []
        self.step_id_2_interpolated_bid = None

    def add_price_quantity_pair(self, price, quantity):
        """Adds price-quantity tuples to the bid"""
        self.price_quantity_pairs.append(PriceQuantityPair(p=price, q=quantity))

    def set_price_quantity_pairs(self, price_quantity_pairs):
        """Sets the price-quantity pairs"""
        self.price_quantity_pairs = price_quantity_pairs

    def print_bid(self):
        pass


class StepHourlyBid(Bid):

    """Implements a step hourly bid"""

    def __init__(self, bid_id, period):
        num_period = 1
        super(StepHourlyBid, self).__init__(bid_id, num_period)
        self.period = period
        self.price_quantity_pairs = []
        self.step_id_2_simple_bid = None

    def add_price_quantity_pair(self, price, quantity):
        """Adds price-quantity tuples to the bid"""
        self.price_quantity_pairs.append(PriceQuantityPair(p=price, q=quantity))

    def set_price_quantity_pairs(self, price_quantity_pairs):
        """Sets the price-quantity pairs"""
        self.price_quantity_pairs = price_quantity_pairs

    def print_bid(self):
        pass


class BlockBid(Bid):

    def __init__(self, bid_id, num_period, period, price, quantity, link):
        super(BlockBid, self).__init__(bid_id, num_period)
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
        super(FlexibleBid, self).__init__(bid_id, num_period)
        self.price = price
        self.quantity = quantity

    def print_bid(self):
        pass


class BidType(Enum):

    STEP_HOURLY = 'S'
    BLOCK = 'B'
    FLEXIBLE = 'F'
    PIECEWISE_HOURLY = 'P'


class InputStats:

    def __init__(self, dam_data):
        self._average_block_bid_num_period = np.mean(
            [block_bid.num_period for block_bid in dam_data.dam_bids.bid_id_2_block_bid.values()])
        self._average_block_bid_quantity = np.mean(
            [abs(block_bid.quantity) for block_bid in dam_data.dam_bids.bid_id_2_block_bid.values()])
        self._number_of_step_hourly_bids = len(dam_data.dam_bids.bid_id_2_step_hourly_bid)
        self._number_of_piecewise_hourly_bids = len(dam_data.dam_bids.bid_id_2_piecewise_hourly_bid)
        self._number_of_block_bids = len(dam_data.dam_bids.bid_id_2_block_bid)
        self._number_of_flexible_bids = len(dam_data.dam_bids.bid_id_2_flexible_bid)
    
    def average_block_bid_num_period(self):
        """Returns the average block bid number of periods"""
        return self._average_block_bid_num_period

    def average_block_bid_quantity(self):
        """Returns the average block bid quantity"""
        return self._average_block_bid_quantity

    def number_of_step_hourly_bids(self):
        """Returns the number of step hourly bids"""
        return self._number_of_step_hourly_bids

    def number_of_piecewise_hourly_bids(self):
        """Returns the number of piecewise hourly bids"""
        return self._number_of_piecewise_hourly_bids

    def number_of_block_bids(self):
        """Returns the number of block bids"""
        return self._number_of_block_bids

    def number_of_flexible_bids(self):
        """Returns the number of flexible bids"""
        return self._number_of_flexible_bids

    def print_stats(self):
        logger.info('Printing input stats...')
        logger.info('Number of step hourly bids: %s', self._number_of_step_hourly_bids)
        logger.info('Number of piecewise linear hourly bids: %s', self._number_of_piecewise_hourly_bids)
        logger.info('Number of block bids: %s', self._number_of_block_bids)
        logger.info('Number of flexible bids: %s', self._number_of_flexible_bids)
