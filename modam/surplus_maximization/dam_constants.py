#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:36:48 2018

@author: gokhanceyhan
"""

DAM_DATA_BID_ID_HEADER = "bid_id"
DAM_DATA_BID_PERIOD_HEADER = "period"
DAM_DATA_BID_BUCKET_ID_HEADER = "bucket_id"
DAM_DATA_BID_TYPE_HEADER = "bid_type"
DAM_DATA_BID_ZONE_HEADER = "zone"
DAM_DATA_BID_QUANTITY_HEADER = "quantity"
DAM_DATA_BID_PRICE_HEADER = "price"
DAM_DATA_BID_NUMBER_OF_PERIODS_HEADER = "num_periods" 
DAM_DATA_BID_LINK_HEADER = "link"
DAM_DATA_BID_END_PERIOD_HEADER = "end_period"
DAM_DATA_BID_EXCLUSIVE_GROUP_ID_HEADER = "exclusive_group_id"
OBJ_COMP_TOL = 1  # if the difference between two obj values is less than this, we assume they are equal.
PAB_PRB_SURPLUS_TOL = 0.1  #  bid is PAB/PRB if s < - PAB_PRB_SURPLUS_TOL
PRICE_COMP_TOL = 0.1  # if the difference between two price values is less than this, we assume they are equal.
