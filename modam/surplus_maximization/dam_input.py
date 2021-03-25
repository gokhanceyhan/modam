"""
Created on Thu Aug  2 21:07:42 2018

@author: gokhanceyhan
"""

from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from enum import Enum
import logging
import numpy as np
import pandas as pd

import modam.surplus_maximization.dam_constants as dc
from modam.surplus_maximization.dam_exceptions import InvalidBidException
from modam.surplus_maximization.dam_utils import interpolate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PriceQuantityPair = namedtuple("PriceQuantityPair", ["p", "q"])
SimpleBid = namedtuple("SimpleBid", ["p", "q"])
InterpolatedBid = namedtuple("InterpolatedBid", ["p_start", "p_end", "q"])


class BidType(Enum):

    STEP_HOURLY = 'S'
    BLOCK = 'B'
    FLEXIBLE = 'F'
    PIECEWISE_HOURLY = 'P'
    PROFILE_BLOCK = 'PB'
    PROFILE_FLEXIBLE = 'PF'


class ConstraintType(Enum):

    EQUAL_TO = "=="
    GREATER_THAN_EQUAL_TO = ">="
    LESS_THAN_EQUAL_TO = "<="


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
        "link": str,
        "end_period": "int16", 
        "exclusive_group_id": str}
    _FLEXIBLE_BID_CONVERTED_ID_TEMPLATE = "{id}_{period}"

    # minimum allowable bid price
    MIN_PRICE = 0
    # maximum allowable bid price
    MAX_PRICE = 2000
    # number of time periods in dam
    NUM_PERIODS = 24
    # reading the dam input from the file specified

    def __init__(self):
        self.dam_original_bids = None
        self.dam_bids = None
        self.block_bid_id_2_linked_block_bid_id = None
        self.block_bid_id_2_child_block_bids = None
        self.exclusive_group_id_2_block_bid_ids = None
        self.block_bid_constraints_matrix = None
        self.block_bid_constraints_bid_ids = None
        self.block_bid_constraints_rhs = None
        self.block_bid_constraints_types = None

    @staticmethod
    def _convert_flexible_bids_to_block_bids(bid_id_2_flexible_bid):
        """Converts flexible bids to mutually exclusive block bids"""
        bid_id_2_block_bid = {}
        for bid_id, flexible_bid in bid_id_2_flexible_bid.items():
            start_period = flexible_bid.period
            end_period = flexible_bid.end_period
            num_period = flexible_bid.num_period
            price = flexible_bid.price
            quantities = flexible_bid.quantities
            for period in range(start_period, end_period - num_period + 2):
                block_bid_id = DamData._FLEXIBLE_BID_CONVERTED_ID_TEMPLATE.format(id=bid_id, period=period)
                block_bid = BlockBid(
                    block_bid_id, num_period, period, price, exclusive_group_id=bid_id, from_flexible=True)
                for tidx, q in enumerate(quantities):
                    block_bid.insert_quantity(period + tidx, q)
                bid_id_2_block_bid[block_bid_id] = block_bid
        return bid_id_2_block_bid

    def _create_block_bid_constraints(self):
        """Creates the constraints for the linked and mututally exclusive block bids"""
        # use the block bids from 'dam_bids' as it contains the block bids generated from the flexible bids
        block_bids = self.dam_bids.bid_id_2_block_bid.values()
        block_bid_ids = [block_bid.bid_id for block_bid in block_bids]
        block_bid_id_2_index = {block_bid_id: index for index, block_bid_id in enumerate(block_bid_ids)}
        exclusive_group_id_2_block_bid_ids = defaultdict(list)
        block_bid_id_2_linked_block_bid_id = {}
        block_bid_id_2_child_block_bids = defaultdict(list)
        for block_bid in block_bids:
            if block_bid.link is not None:
                parent_bid_id = block_bid.link
                block_bid_id_2_linked_block_bid_id[block_bid.bid_id] = parent_bid_id
                block_bid_id_2_child_block_bids[parent_bid_id].append(block_bid)
            if block_bid.exclusive_group_id is not None:
                exclusive_group_id_2_block_bid_ids[block_bid.exclusive_group_id].append(block_bid.bid_id)
        self.block_bid_id_2_linked_block_bid_id = block_bid_id_2_linked_block_bid_id
        self.block_bid_id_2_child_block_bids = block_bid_id_2_child_block_bids
        self.exclusive_group_id_2_block_bid_ids = exclusive_group_id_2_block_bid_ids
        # create the constraint matrix and rhs
        num_variables = len(block_bid_ids)
        num_constraints = len(exclusive_group_id_2_block_bid_ids) + len(block_bid_id_2_linked_block_bid_id)
        matrix = np.zeros((num_constraints, num_variables))
        rhs = np.zeros(num_constraints)
        cidx = 0
        for group_id, block_bid_ids_in_group in exclusive_group_id_2_block_bid_ids.items():
            rhs[cidx] = 1
            for block_bid_id in block_bid_ids_in_group:
                vidx = block_bid_id_2_index[block_bid_id]
                matrix[cidx, vidx] = 1
            cidx += 1
        for block_bid_id, linked_block_bid_id in block_bid_id_2_linked_block_bid_id.items():
            block_bid_vidx = block_bid_id_2_index[block_bid_id]
            linked_block_bid_vidx = block_bid_id_2_index[linked_block_bid_id]
            matrix[cidx, block_bid_vidx] = 1
            matrix[cidx, linked_block_bid_vidx] = -1
            cidx += 1
        self.block_bid_constraints_bid_ids = block_bid_ids
        self.block_bid_constraints_matrix = matrix
        self.block_bid_constraints_rhs = rhs
        self.block_bid_constraints_types = np.full(num_constraints, ConstraintType.LESS_THAN_EQUAL_TO)

    @staticmethod
    def _create_idential_bid_lists(bids_df):
        """Create the list of identical bid sets for different bid types
        
        NOTE: Currently, only block bids are checked."""
        bid_type_2_identical_bid_lists = {}
        block_bids_df = bids_df[
            (bids_df["bid_type"] == BidType.BLOCK.value) & (bids_df["link"].isnull())].reset_index(drop=True)
        exclusive_group_column_exists = dc.DAM_DATA_BID_EXCLUSIVE_GROUP_ID_HEADER in bids_df.columns
        if exclusive_group_column_exists:
            block_bids_df = block_bids_df[block_bids_df["exclusive_group_id"].isnull()]
        bid_lists = []
        for name, df_ in block_bids_df.groupby(by=["period", "num_periods", "quantity", "price"]):
            if df_["bid_id"].count() == 1:
                continue
            sorted_df = df_.sort_values("bid_id")
            identical_bids = []
            for _, row in sorted_df.iterrows():
                bid_type = row[dc.DAM_DATA_BID_TYPE_HEADER]
                bid_id = row[dc.DAM_DATA_BID_ID_HEADER]
                num_periods = row[dc.DAM_DATA_BID_NUMBER_OF_PERIODS_HEADER]
                period = row[dc.DAM_DATA_BID_PERIOD_HEADER]
                price = row[dc.DAM_DATA_BID_PRICE_HEADER]
                quantity = row[dc.DAM_DATA_BID_QUANTITY_HEADER]
                link = row[dc.DAM_DATA_BID_LINK_HEADER]
                block_bid = BlockBid(bid_id, num_periods, period, price, link)
                for t in range(period, period + num_periods):
                    block_bid.insert_quantity(t, quantity)
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
        """Reads input data in the specified path and creates the bid set"""
        # read data as a data frame
        df = pd.read_csv(file_path, dtype=DamData._FIELD_NAME_2_DATA_TYPE)
        df = df.iloc[:, 4:]
        end_period_column_exists = dc.DAM_DATA_BID_END_PERIOD_HEADER in df.columns
        exclusive_group_column_exists = dc.DAM_DATA_BID_EXCLUSIVE_GROUP_ID_HEADER in df.columns
        # define bid maps
        bid_id_2_step_hourly_bid = {}
        bid_id_2_piecewise_hourly_bid = {}
        bid_id_2_block_bid = {}
        bid_id_2_flexible_bid = {}
        # iterate data frame and create bid objects
        for index, row in df.iterrows():
            bid_type = row[dc.DAM_DATA_BID_TYPE_HEADER]
            bid_id = row[dc.DAM_DATA_BID_ID_HEADER]
            num_periods = row[dc.DAM_DATA_BID_NUMBER_OF_PERIODS_HEADER]
            period = row[dc.DAM_DATA_BID_PERIOD_HEADER]
            price = row[dc.DAM_DATA_BID_PRICE_HEADER]
            quantity = row[dc.DAM_DATA_BID_QUANTITY_HEADER]
            link = row[dc.DAM_DATA_BID_LINK_HEADER]
            link = link if not pd.isnull(link) else None
            # block bid
            if bid_type == BidType.BLOCK.value:
                exclusive_group_id = row[dc.DAM_DATA_BID_EXCLUSIVE_GROUP_ID_HEADER] if exclusive_group_column_exists \
                    else None
                block_bid = BlockBid(
                    bid_id, num_periods, period, price, link=link, exclusive_group_id=exclusive_group_id)
                for t in range(period, period + num_periods):
                    block_bid.insert_quantity(t, quantity)
                bid_id_2_block_bid[bid_id] = block_bid
            # profile block bid
            elif bid_type == BidType.PROFILE_BLOCK.value:
                exclusive_group_id = row[dc.DAM_DATA_BID_EXCLUSIVE_GROUP_ID_HEADER] if exclusive_group_column_exists \
                    else None
                if bid_id not in bid_id_2_block_bid:
                    block_bid = BlockBid(
                        bid_id, num_periods, period, price, link=link, exclusive_group_id=exclusive_group_id)
                    block_bid.insert_quantity(period, quantity)
                    bid_id_2_block_bid[bid_id] = block_bid
                else:
                    bid_id_2_block_bid[bid_id].insert_quantity(period, quantity)
            # flexible bid
            elif bid_type == BidType.FLEXIBLE.value:
                period = period if period else 1
                end_period = row[dc.DAM_DATA_BID_END_PERIOD_HEADER] if end_period_column_exists else DamData.NUM_PERIODS
                flexible_bid = FlexibleBid(bid_id, num_periods, price, period=period, end_period=end_period)
                quantities = [quantity] * num_periods
                flexible_bid.set_quantities(quantities)
                bid_id_2_flexible_bid[bid_id] = flexible_bid
            # profile flexible bid
            elif bid_type == BidType.PROFILE_FLEXIBLE.value:
                if bid_id not in bid_id_2_flexible_bid:
                    period = period if period else 1
                    end_period = row[dc.DAM_DATA_BID_END_PERIOD_HEADER] if end_period_column_exists else \
                        DamData.NUM_PERIOD
                    flexible_bid = FlexibleBid(bid_id, num_periods, price, period=period, end_period=end_period)
                    flexible_bid.add_quantity(quantity)
                    bid_id_2_flexible_bid[bid_id] = flexible_bid
                else:
                    bid_id_2_flexible_bid[bid_id].add_quantity(quantity)
            # step hourly bid
            elif bid_type == BidType.STEP_HOURLY.value:
                if bid_id not in bid_id_2_step_hourly_bid:
                    hourly_bid = StepHourlyBid(bid_id, period)
                    hourly_bid.add_price_quantity_pair(price, quantity)
                    bid_id_2_step_hourly_bid[bid_id] = hourly_bid
                else:
                    bid_id_2_step_hourly_bid[bid_id].add_price_quantity_pair(price, quantity)
            # piecewise hourly bid
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
        dam_bids = DamBids(
            bid_id_2_step_hourly_bid, bid_id_2_piecewise_hourly_bid, bid_id_2_block_bid, bid_id_2_flexible_bid, 
            bid_type_2_identical_bid_lists=bid_type_2_identical_bid_lists)
        # validate the bids
        dam_bids.validate()
        self.dam_original_bids = dam_bids
        # convert flexible bids to block bids for modeling purposes
        updated_bid_id_2_block_bid = bid_id_2_block_bid.copy()
        updated_bid_id_2_block_bid.update(DamData._convert_flexible_bids_to_block_bids(bid_id_2_flexible_bid))
        updated_bid_id_2_flexible_bid = {}
        self.dam_bids = DamBids(
            bid_id_2_step_hourly_bid, bid_id_2_piecewise_hourly_bid, updated_bid_id_2_block_bid, 
            updated_bid_id_2_flexible_bid, bid_type_2_identical_bid_lists=bid_type_2_identical_bid_lists)
        self._create_block_bid_constraints()


class DamBids:

    def __init__(
            self, bid_id_2_step_hourly_bid, bid_id_2_piecewise_hourly_bid, bid_id_2_block_bid, bid_id_2_flexible_bid, 
            bid_type_2_identical_bid_lists=None):
        self.bid_id_2_step_hourly_bid = bid_id_2_step_hourly_bid
        self.bid_id_2_piecewise_hourly_bid = bid_id_2_piecewise_hourly_bid
        self.bid_id_2_block_bid = bid_id_2_block_bid
        self.bid_id_2_flexible_bid = bid_id_2_flexible_bid
        self.bid_type_2_identical_bid_lists = bid_type_2_identical_bid_lists or {}

    def validate(self):
        """Validates the bid set"""
        for hourly_bid in self.bid_id_2_step_hourly_bid.values():
            hourly_bid.validate()
        for hourly_bid in self.bid_id_2_piecewise_hourly_bid.values():
            hourly_bid.validate()
        for block_bid in self.bid_id_2_block_bid.values():
            block_bid.validate()


class Bid(ABC):

    def __init__(self, bid_id, num_period):
        self.bid_id = bid_id
        self.num_period = num_period

    @abstractmethod
    def print_bid(self):
        pass
    
    @abstractmethod
    def validate(self):
        """Validates the bid"""


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
        return self.__dict__

    def validate(self):
        previous_price = self.price_quantity_pairs[0].p
        for price, _ in self.price_quantity_pairs[1:]:
            if price <= previous_price:
                raise InvalidBidException(f"prices must form an increasing sequence at block bid {self.bid_id}")


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
        return self.__dict__

    def validate(self):
        previous_price = self.price_quantity_pairs[0].p
        for price, _ in self.price_quantity_pairs[1:]:
            if price <= previous_price:
                raise InvalidBidException(f"prices must form an increasing sequence at block bid {self.bid_id}")


class BlockBid(Bid):

    def __init__(self, bid_id, num_period, period, price, exclusive_group_id=None, from_flexible=False, link=None):
        super(BlockBid, self).__init__(bid_id, num_period)
        self.period = period
        self.price = price
        self.quantities = [0.0] * DamData.NUM_PERIODS
        self.link = link
        self.exclusive_group_id = exclusive_group_id
        self.from_flexible = from_flexible

    def insert_quantity(self, period, quantity):
        """Inserts the quantity at the specified period"""
        self.quantities[period - 1] = quantity

    @property
    def is_supply(self):
        """Returns True if the block bid is a supply block bid, otherwise False"""
        return self.quantities[self.period - 1] <= 0

    def print_bid(self):
        return self.__dict__
    
    def quantity(self, period):
        """Returns the quantity for the given period"""
        return self.quantities[period - 1]

    def set_quantities(self, quantities):
        """Sets the quantities of the bid"""
        self.quantities = quantities

    @property
    def total_quantity(self):
        """Returns the total quantity of the block bid"""
        return sum(self.quantities)

    def validate(self):
        if not (all([q <= 0 for q in self.quantities]) or all([q >= 0 for q in self.quantities])):
            raise InvalidBidException(
                f"all quantities must be either non-positive or non-negative at block bid {self.bid_id}")


class FlexibleBid(Bid):

    def __init__(self, bid_id, num_period, price, period=1, end_period=24):
        super(FlexibleBid, self).__init__(bid_id, num_period)
        self.price = price
        self.quantities = []
        self.period = period
        self.end_period = end_period

    def add_quantity(self, quantity):
        """Adds the quantity to the list of existing quantities"""
        self.quantities.append(quantity)

    @property
    def is_supply(self):
        """Returns True if the block bid is a supply flexible bid, otherwise False"""
        return self.quantities[0] <= 0

    def print_bid(self):
        return self.__dict__

    def set_quantities(self, quantities):
        """Sets the quantities of the bid"""
        self.quantities = quantities

    @property
    def total_quantity(self):
        """Returns the total quantity of the flexible bid"""
        return sum(self.quantities)

    def validate(self):
        if not (all([q <= 0 for q in self.quantities]) or all([q >= 0 for q in self.quantities])):
            raise InvalidBidException(
                f"all quantities must be either non-positive or non-negative at flexible bid {self.bid_id}")
        if self.period + self.num_period > self._end_period:
            raise InvalidBidException(
                f"end period of the flexible bid must at least be equal to the sum of the start period and the " \
                "number of periods of fliexible bid {self.bid_id}")


class InputStats:

    def __init__(self, dam_data):
        bids = dam_data.dam_original_bids
        self._average_block_bid_num_period = np.mean(
            [block_bid.num_period for block_bid in bids.bid_id_2_block_bid.values()])
        self._average_block_bid_quantity = np.mean(
            [abs(block_bid.total_quantity / block_bid.num_period) for block_bid in bids.bid_id_2_block_bid.values()])
        self._number_of_step_hourly_bids = len(bids.bid_id_2_step_hourly_bid)
        self._number_of_piecewise_hourly_bids = len(bids.bid_id_2_piecewise_hourly_bid)
        self._number_of_block_bids = len(bids.bid_id_2_block_bid)
        self._number_of_flexible_bids = len(bids.bid_id_2_flexible_bid)
        self._number_of_linked_block_bids = len(
            [block_bid for block_bid in bids.bid_id_2_block_bid.values() if block_bid.link])
    
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

    def number_of_linked_block_bids(self):
        """Returns the number of linked block bids"""
        return self._number_of_linked_block_bids

    def print_stats(self):
        logger.info('Printing input stats...')
        logger.info('Number of step hourly bids: %s', self._number_of_step_hourly_bids)
        logger.info('Number of piecewise linear hourly bids: %s', self._number_of_piecewise_hourly_bids)
        logger.info('Number of block bids: %s', self._number_of_block_bids)
        logger.info('Number of flexible bids: %s', self._number_of_flexible_bids)
        logger.info('Number of linked block bids: %s', self._number_of_linked_block_bids)
