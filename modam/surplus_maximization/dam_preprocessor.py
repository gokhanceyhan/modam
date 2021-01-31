"""Implements the pre-processing utilities for the day-ahead market clearin problem solver"""

from collections import namedtuple

import os
import pandas as pd

from modam.surplus_maximization.dam_input import SimpleBid, StepHourlyBid


class Preprocessor:

    """Implements pre-processing functions for data-cleansing and problem reduction"""

    SimpleBid = namedtuple("SimpleBid", ["p", "q", "t"])

    def __init__(self, dam_data, working_dir):
        self._dam_data = dam_data
        self._working_dir = working_dir

    def _create_aggregate_hourly_bids_from_step_hourly_bids(self, approximate=False):
        """Creates aggregate supply and demand hourly bids for each period from step hourly bids"""
        bid_id_2_hourly_bid = self._dam_data.dam_bids.bid_id_2_step_hourly_bid
        records = []
        for bid_id, hourly_bid in bid_id_2_hourly_bid.items():
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                records.append({'p': simple_bid.p, 'q': simple_bid.q, 't': hourly_bid.period})
        if not records:
            return {}
        hourly_bids_df = pd.DataFrame.from_records(records)
        # drop zero-quantity bids
        hourly_bids_df = hourly_bids_df[hourly_bids_df["q"] != 0.0]
        hourly_bids_df["d"] = hourly_bids_df["q"].apply(lambda x: "supply" if x < 0 else "demand")
        # aggregate
        bid_id_2_aggregated_hourly_bid = {}
        aggr_hourly_bids_df = hourly_bids_df.groupby(["t", "d", "p"]).sum().reset_index()
        if approximate:
            aggr_hourly_bids_df = self._group_hourly_bids(aggr_hourly_bids_df)
        for (t, d), group in aggr_hourly_bids_df.groupby(["t", "d"]):
            bid_id = "{}_{}".format(t, d)
            hourly_bid = StepHourlyBid(bid_id, t)
            df_ = group.reset_index()
            hourly_bid.step_id_2_simple_bid = {
                index + 1: SimpleBid(p=row["p"], q=row["q"]) for index, row in df_.iterrows()}
            bid_id_2_aggregated_hourly_bid[bid_id] = hourly_bid
        return bid_id_2_aggregated_hourly_bid

    def _group_hourly_bids(self, df, num_bins=10):
        """Bins the hourly bids with similar prices"""
        df["bin"] = pd.qcut(df["p"], q=num_bins)
        df["p"] = df["bin"].apply(lambda x: x.right)
        df = df.groupby(["t", "d", "p"]).sum().reset_index()
        return df

    def run(self, approximate=False):
        """Runs the pre-preocessor and returns the preprocessed dam data"""
        dam_data = self._dam_data
        dam_bids = dam_data.dam_bids
        bid_id_2_aggregated_hourly_bid = self._create_aggregate_hourly_bids_from_step_hourly_bids(
            approximate=approximate)
        dam_bids.bid_id_2_step_hourly_bid = bid_id_2_aggregated_hourly_bid
        return dam_data
