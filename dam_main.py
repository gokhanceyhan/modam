#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:51:21 2018

@author: gokhanceyhan

Script to run dam clearing software
"""

from dam_input import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # create dam input data
    logger.info('Start reading the input')
    dam_data = DamData()
    dam_data.read_input('../dam_data/data.csv')
    
    # create and print input data stats
    dam_data_stats = InputStats(dam_data)
    dam_data_stats.print_stats()
    

if __name__ == "__main__":
    main()