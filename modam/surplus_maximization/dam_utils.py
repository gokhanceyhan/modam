"""Implements the utility functions for the day-ahead market solver"""

import csv
import numpy as np
import os

import modam.surplus_maximization.dam_constants as dc


def interpolate(p_start, q_start, p_end, q_end, p=None, q=None):
    assert p is not None or q is not None, "either 'p' or 'q' must be given to determine the interpolated point"
    if p is not None:
        assert p_start != p_end
        w = (p - p_start) / (p_end - p_start)
        return q_start + (q_end - q_start) * w
    assert q_start != q_end
    w = (q - q_start) / (q_end - q_start)
    return p_start + (p_end - p_start) * w

def is_accepted_block_bid_pab(bid, mcp):
    first = bid.period - 1
    last = bid.period+bid.num_period-1
    mcp_block = mcp[first:last]
    mcp_avg = sum(mcp_block)/len(mcp_block)
    if bid.is_supply:
        return bid.price > mcp_avg + dc.PRICE_COMP_TOL
    else:
        return bid.price < mcp_avg - dc.PRICE_COMP_TOL


def is_rejected_block_bid_prb(bid, mcp):
    first = bid.period - 1
    last = bid.period+bid.num_period-1
    mcp_block = mcp[first:last]
    mcp_avg = sum(mcp_block)/len(mcp_block)
    if bid.is_supply:
        return bid.price < mcp_avg - dc.PRICE_COMP_TOL
    else:
        return bid.price > mcp_avg + dc.PRICE_COMP_TOL


def calculate_bigm_for_block_bid_loss(block_bid, max_price=2000, min_price=0):
    if block_bid.is_supply:
        return abs((block_bid.price - min_price) * block_bid.quantity * block_bid.num_period)
    # if demand
    return abs((max_price - block_bid.price) * block_bid.quantity * block_bid.num_period)


def calculate_bigm_for_block_bid_missed_surplus(block_bid, max_price=2000, min_price=0):
    if block_bid.is_supply:
        return abs((max_price - block_bid.price) * block_bid.quantity * block_bid.num_period)
    # if demand
    return abs((block_bid.price - min_price) * block_bid.quantity * block_bid.num_period)


def do_block_bids_have_common_period(this_block_bid, that_block_bid):
    this_block_bid_periods = range(this_block_bid.period, this_block_bid.period + this_block_bid.num_period, 1)
    that_block_bid_periods = range(that_block_bid.period, that_block_bid.period + that_block_bid.num_period, 1)
    return len(set(this_block_bid_periods).intersection(set(that_block_bid_periods))) > 0


def find_pabs(market_clearing_prices, accepted_block_bid_ids, bid_id_2_block_bid):
    pabs = []
    for accepted_block_bid_id in accepted_block_bid_ids:
        accepted_block_bid = bid_id_2_block_bid[accepted_block_bid_id]
        if is_accepted_block_bid_pab(accepted_block_bid, market_clearing_prices):
            pabs.append(accepted_block_bid)
    return pabs


def find_prbs(market_clearing_prices, rejected_block_bid_ids, bid_id_2_block_bid):
    prbs = []
    for rejected_block_bid_id in rejected_block_bid_ids:
        rejected_block_bid = bid_id_2_block_bid[rejected_block_bid_id]
        if is_rejected_block_bid_prb(rejected_block_bid, market_clearing_prices):
            prbs.append(rejected_block_bid)
    return prbs


def create_gcut_for_pab(pab, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var):

    if pab.is_supply:
        variables, coefficients, rhs = create_gcut_for_supply_pab(
            pab, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var)
    else:
        variables, coefficients, rhs = create_gcut_for_demand_pab(
            pab, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var)
    return variables, coefficients, rhs


def create_gcut_for_prb(prb, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var):

    if prb.is_supply:
        variables, coefficients, rhs = create_gcut_for_supply_prb(
            prb, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var)
    else:
        variables, coefficients, rhs = create_gcut_for_demand_prb(
            prb, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var)
    return variables, coefficients, rhs


def create_gcut_for_supply_pab(
        pab, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var):
    variables = [bid_id_2_bbid_var[pab.bid_id]]
    coefficients = [-1]
    rhs = 0
    accepted_supply_block_bid_ids = [
        bid_id for bid_id in accepted_block_bid_ids if bid_id != pab.bid_id and bid_id_2_block_bid[bid_id].is_supply]
    rejected_demand_block_bid_ids = [
        bid_id for bid_id in rejected_block_bid_ids if bid_id != pab.bid_id and 
        not bid_id_2_block_bid[bid_id].is_supply]
    # find intersecting accepted supply block bids
    for bbid_id in accepted_supply_block_bid_ids:
        block_bid = bid_id_2_block_bid[bbid_id]
        if do_block_bids_have_common_period(pab, block_bid):
            variables.append(bid_id_2_bbid_var[bbid_id])
            coefficients.append(-1)
            rhs -= 1
    # find intersecting rejected demand block bids
    for bbid_id in rejected_demand_block_bid_ids:
        block_bid = bid_id_2_block_bid[bbid_id]
        if do_block_bids_have_common_period(pab, block_bid):
            variables.append(bid_id_2_bbid_var[bbid_id])
            coefficients.append(1)
    return variables, coefficients, rhs


def create_gcut_for_supply_prb(
        prb, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var):
    variables = [bid_id_2_bbid_var[prb.bid_id]]
    coefficients = [1]
    rhs = 1
    rejected_supply_block_bid_ids = [
        bid_id for bid_id in rejected_block_bid_ids if bid_id != prb.bid_id and bid_id_2_block_bid[bid_id].is_supply]
    accepted_demand_block_bid_ids = [
        bid_id for bid_id in accepted_block_bid_ids if bid_id != prb.bid_id and 
        not bid_id_2_block_bid[bid_id].is_supply]
    # find intersecting rejected supply block bids
    for bbid_id in rejected_supply_block_bid_ids:
        block_bid = bid_id_2_block_bid[bbid_id]
        if do_block_bids_have_common_period(prb, block_bid):
            variables.append(bid_id_2_bbid_var[bbid_id])
            coefficients.append(1)

    # find intersecting accepted demand block bids
    for bbid_id in accepted_demand_block_bid_ids:
        block_bid = bid_id_2_block_bid[bbid_id]
        if do_block_bids_have_common_period(prb, block_bid):
            variables.append(bid_id_2_bbid_var[bbid_id])
            coefficients.append(-1)
            rhs -= 1
    return variables, coefficients, rhs


def create_gcut_for_demand_pab(
        pab, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var):
    variables = [bid_id_2_bbid_var[pab.bid_id]]
    coefficients = [-1]
    rhs = 0
    accepted_demand_block_bid_ids = [
        bid_id for bid_id in accepted_block_bid_ids if bid_id != pab.bid_id and 
        not bid_id_2_block_bid[bid_id].is_supply]
    rejected_supply_block_bid_ids = [
        bid_id for bid_id in rejected_block_bid_ids if bid_id != pab.bid_id and bid_id_2_block_bid[bid_id].is_supply]
    # find intersecting accepted demand block bids
    for bbid_id in accepted_demand_block_bid_ids:
        block_bid = bid_id_2_block_bid[bbid_id]
        if do_block_bids_have_common_period(pab, block_bid):
            variables.append(bid_id_2_bbid_var[bbid_id])
            coefficients.append(-1)
            rhs -= 1
    # find intersecting rejected supply block bids
    for bbid_id in rejected_supply_block_bid_ids:
        block_bid = bid_id_2_block_bid[bbid_id]
        if do_block_bids_have_common_period(pab, block_bid):
            variables.append(bid_id_2_bbid_var[bbid_id])
            coefficients.append(1)
    return variables, coefficients, rhs


def create_gcut_for_demand_prb(
        prb, accepted_block_bid_ids, rejected_block_bid_ids, bid_id_2_block_bid, bid_id_2_bbid_var):
    variables = [bid_id_2_bbid_var[prb.bid_id]]
    coefficients = [1]
    rhs = 1
    rejected_demand_block_bid_ids = [
        bid_id for bid_id in rejected_block_bid_ids if bid_id != prb.bid_id and 
        not bid_id_2_block_bid[bid_id].is_supply]
    accepted_supply_block_bid_ids = [
        bid_id for bid_id in accepted_block_bid_ids if bid_id != prb.bid_id and bid_id_2_block_bid[bid_id].is_supply]
    # find intersecting rejected demand block bids
    for bbid_id in rejected_demand_block_bid_ids:
        block_bid = bid_id_2_block_bid[bbid_id]
        if do_block_bids_have_common_period(prb, block_bid):
            variables.append(bid_id_2_bbid_var[bbid_id])
            coefficients.append(1)
    # find intersecting accepted supply block bids
    for bbid_id in accepted_supply_block_bid_ids:
        block_bid = bid_id_2_block_bid[bbid_id]
        if do_block_bids_have_common_period(prb, block_bid):
            variables.append(bid_id_2_bbid_var[bbid_id])
            coefficients.append(-1)
            rhs -= 1
    return variables, coefficients, rhs


def get_pab_info(bid, mcp):
    """Returns (True, price_gap, loss) tuple if the bid is a pab, otherwise (False, 0, 0)"""
    first = bid.period - 1
    last = bid.period+bid.num_period-1
    mcp_block = mcp[first:last]
    mcp_avg = sum(mcp_block)/len(mcp_block)
    if bid.is_supply:
        if bid.price > mcp_avg + dc.PRICE_COMP_TOL:
            return True, bid.price - mcp_avg, (mcp_avg - bid.price) * bid.quantity * len(mcp_block)
        else:
            return False, 0, 0
    else:
        if bid.price < mcp_avg - dc.PRICE_COMP_TOL:
            return True, mcp_avg - bid.price, (mcp_avg - bid.price) * bid.quantity * len(mcp_block)
        else:
            return False, 0, 0


def get_prb_info(bid, mcp):
    """Returns (True, price_gap, missed_surplus) tuple if the bid is a prb, otherwise (False, 0, 0)"""
    first = bid.period - 1
    last = bid.period+bid.num_period-1
    mcp_block = mcp[first:last]
    mcp_avg = sum(mcp_block)/len(mcp_block)
    if bid.is_supply:
        if bid.price < mcp_avg - dc.PRICE_COMP_TOL:
            return True, mcp_avg - bid.price, (bid.price - mcp_avg) * bid.quantity * len(mcp_block)
        else:
            return False, 0, 0
    else:
        if bid.price > mcp_avg + dc.PRICE_COMP_TOL:
            return True, bid.price - mcp_avg, (bid.price - mcp_avg) * bid.quantity * len(mcp_block)
        else:
            return False, 0, 0


def generate_market_result_statistics(dam_bids, dam_solution):
    """Generates the market result statistics, update the solution, and returns it"""
    # find the set of PABs and PRBs
    pabs = []
    prbs = []
    loss = 0
    missed_surplus = 0
    average_pab_price_gap = 0
    pab_with_max_price_gap = None
    average_prb_price_gap = 0
    prb_with_max_price_gap = None

    for bid_id in dam_solution.accepted_block_bids:
        bid = dam_bids.bid_id_2_block_bid[bid_id]
        pab, price_gap_, loss_ = get_pab_info(bid, dam_solution.market_clearing_prices)
        if not pab:
            continue
        pabs.append(bid_id)
        loss += loss_
        average_pab_price_gap += price_gap_
        if not pab_with_max_price_gap or pab_with_max_price_gap[2] < price_gap_:
            pab_with_max_price_gap = (abs(bid.quantity), bid.num_period, price_gap_)
    average_pab_price_gap = (average_pab_price_gap / len(pabs)) if pabs else None

    for bid_id in dam_solution.rejected_block_bids:
        bid = dam_bids.bid_id_2_block_bid[bid_id]
        prb, price_gap_, missed_surplus_ = get_prb_info(bid, dam_solution.market_clearing_prices)
        if not prb:
            continue
        prbs.append(bid_id)
        missed_surplus += missed_surplus_
        average_prb_price_gap += price_gap_
        if not prb_with_max_price_gap or prb_with_max_price_gap[2] < price_gap_:
            prb_with_max_price_gap = (abs(bid.quantity), bid.num_period, price_gap_)
    average_prb_price_gap = (average_prb_price_gap / len(prbs)) if prbs else None

    # update the solution
    dam_solution.loss = loss
    dam_solution.missed_surplus = missed_surplus
    dam_solution.num_pab = len(pabs)
    dam_solution.num_prb = len(prbs)
    dam_solution.average_pab_price_gap = average_pab_price_gap
    dam_solution.average_prb_price_gap = average_prb_price_gap
    dam_solution.max_pab_price_gap = pab_with_max_price_gap[2] if pab_with_max_price_gap else None
    dam_solution.max_prb_price_gap = prb_with_max_price_gap[2] if prb_with_max_price_gap else None
    dam_solution.num_periods_for_pab_with_max_price_gap = pab_with_max_price_gap[1] if pab_with_max_price_gap else None
    dam_solution.num_periods_for_prb_with_max_price_gap = prb_with_max_price_gap[1] if prb_with_max_price_gap else None
    dam_solution.quantity_pab_with_max_price_gap = pab_with_max_price_gap[0] if pab_with_max_price_gap else None
    dam_solution.quantity_prb_with_max_price_gap = prb_with_max_price_gap[0] if prb_with_max_price_gap else None
    return dam_solution


def write_runners_to_file(runners, working_dir):
    file_path = os.path.join(working_dir, "tests.csv")
    with open(file_path, mode='w') as csv_file:
        fieldnames = [
            'problem file', 'problem type', 'solver', 'method', 'time limit', 'relative gap tolerance', 
            'step hourly bids', 'piecewise hourly bids', 
            'block bids', 'flexible bids', 'valid', 'total surplus', 'solver status', 'best bound', 'relative gap',
            'elapsed solver time', 'number of solutions', 'number of nodes', 'number of subproblems', 
            'number of user cuts', 'avg block bid num period', 'avg block bid quantity', 'num accepted block bids', 
            'num rejected block bids', 'avg. mcp', 'num pab', 'num prb', 'loss', 'missed surplus', 
            'avg pab price gap', 'max pab price gap', 'num periods for pab with max price gap', 
            'quantity pab with max price gap', 'avg prb price gap', 'max prb price gap', 
            'num periods for prb with max price gap', 'quantity prb with max price gap']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for runner in runners:
            input_stats = runner.input_stats()
            optimization_status = runner.output().optimization_status()
            optimization_stats = runner.output().optimization_stats()
            benders_decomposition_stats = optimization_stats.benders_decomposition_stats()
            dam_solution = runner.output().dam_solution()
            row = {
                'problem file': runner.input_file_name(),
                'problem type': runner.problem_type().value,
                'solver': runner.solver().value,
                'method': runner.method().value,
                'time limit': runner.time_limit(),
                'relative gap tolerance': runner.relative_gap_tolerance(),
                'step hourly bids': input_stats.number_of_step_hourly_bids(),
                'piecewise hourly bids': input_stats.number_of_piecewise_hourly_bids(),
                'block bids': input_stats.number_of_block_bids(),
                'flexible bids': input_stats.number_of_flexible_bids(),
                'valid': False if dam_solution is None else dam_solution.is_valid,
                'solver status': optimization_status.solver_status(),
                'best bound': optimization_status.best_bound(),
                'relative gap': optimization_status.relative_gap(),
                'elapsed solver time': optimization_stats.elapsed_time(),
                'number of solutions': optimization_stats.number_of_solutions(),
                'number of nodes': optimization_stats.number_of_nodes(),
                'number of subproblems': benders_decomposition_stats.number_of_subproblems_solved(),
                'number of user cuts': benders_decomposition_stats.number_of_user_cuts_added(),
                'avg block bid num period': input_stats.average_block_bid_num_period(),
                'avg block bid quantity': input_stats.average_block_bid_quantity()    
            }
            if dam_solution:
                row.update(
                    {
                        'total surplus': dam_solution.total_surplus,
                        'num accepted block bids': len(dam_solution.accepted_block_bids),
                        'num rejected block bids': len(dam_solution.rejected_block_bids),
                        'avg. mcp': np.mean(dam_solution.market_clearing_prices),
                        'num pab': dam_solution.num_pab, 
                        'num prb': dam_solution.num_prb, 
                        'loss': dam_solution.loss, 
                        'missed surplus': dam_solution.missed_surplus, 
                        'avg pab price gap': dam_solution.average_pab_price_gap, 
                        'max pab price gap': dam_solution.max_pab_price_gap, 
                        'num periods for pab with max price gap': dam_solution.num_periods_for_pab_with_max_price_gap, 
                        'quantity pab with max price gap': dam_solution.quantity_pab_with_max_price_gap,
                        'avg prb price gap': dam_solution.average_prb_price_gap, 
                        'max prb price gap': dam_solution.max_prb_price_gap, 
                        'num periods for prb with max price gap': dam_solution.num_periods_for_prb_with_max_price_gap, 
                        'quantity prb with max price gap': dam_solution.quantity_prb_with_max_price_gap
                    }
                )
            writer.writerow(row)
