"""Implements the utility functions for the day-ahead market solver"""

import csv
import os

import modam.surplus_maximization.dam_constants as dc


def create_simple_bids_from_hourly_bid(hourly_bid):
    step_id_2_simple_bid = {}
    count = 1
    prev_qnt = 0
    if hourly_bid.price_quantity_pairs[0][1] <= 0:
        # supply bid
        for price, quantity in hourly_bid.price_quantity_pairs:
            step_id_2_simple_bid[count] = (price, quantity - prev_qnt)
            prev_qnt = quantity
            count += 1
    elif hourly_bid.price_quantity_pairs[len(hourly_bid.price_quantity_pairs)-1][1] >= 0:
        # demand bid
        for price, quantity in reversed(hourly_bid.price_quantity_pairs):
            step_id_2_simple_bid[count] = (price, quantity - prev_qnt)
            prev_qnt = quantity
            count += 1
    else:
        prev_qnt = hourly_bid.price_quantity_pairs[0][1]
        prev_prc = hourly_bid.price_quantity_pairs[0][0]
        for price, quantity in hourly_bid.price_quantity_pairs[1:]:
            if quantity >= 0:
                # demand step
                step_id_2_simple_bid[count] = (prev_prc, prev_qnt - quantity)
                prev_prc = price
            elif prev_qnt > 0:
                # first supply step
                step_id_2_simple_bid[count] = (prev_prc, prev_qnt)
                count += 1
                step_id_2_simple_bid[count] = (price, quantity)
            else:
                # supply step
                step_id_2_simple_bid[count] = (price, quantity - prev_qnt)
            prev_qnt = quantity
            count += 1
    return {step_id: simple_bid for step_id, simple_bid in step_id_2_simple_bid.items() if abs(simple_bid[1]) > 0}


def is_accepted_block_bid_pab(bid, mcp):
    first = bid.period - 1
    last = bid.period+bid.num_period-1
    mcp_block = mcp[first:last]
    mcp_avg = sum(mcp_block)/len(mcp_block)
    if bid.is_supply:
        if bid.price > mcp_avg + dc.PRICE_COMP_TOL:
            return True
        else:
            return False
    else:
        if bid.price < mcp_avg - dc.PRICE_COMP_TOL:
            return True
        else:
            return False


def is_rejected_block_bid_prb(bid, mcp):
    first = bid.period - 1
    last = bid.period+bid.num_period-1
    mcp_block = mcp[first:last]
    mcp_avg = sum(mcp_block)/len(mcp_block)
    if bid.is_supply:
        if bid.price < mcp_avg - dc.PRICE_COMP_TOL:
            return True
        else:
            return False
    else:
        if bid.price > mcp_avg + dc.PRICE_COMP_TOL:
            return True
        else:
            return False


def calculate_bigm_for_block_bid_loss(block_bid):
    return abs(block_bid.price * block_bid.quantity)


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


def write_runners_to_file(runners, working_dir):
    file_path = os.path.join(working_dir, "tests.csv")
    with open(file_path, mode='w') as csv_file:
        fieldnames = [
            'problem file', 'problem type', 'solver', 'method', 'time limit', 'relative gap tolerance', 'hourly bids',
            'block bids', 'flexible bids', 'valid', 'total surplus', 'solver status', 'best bound', 'relative gap',
            'elapsed solver time', 'number of solutions', 'number of nodes', 'number of subproblems',
            'number of user cuts']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for runner in runners:
            writer.writerow(
                {
                    'problem file': runner.input_file_name(),
                    'problem type': runner.problem_type().value,
                    'solver': runner.solver().value,
                    'method': runner.method().value,
                    'time limit': runner.time_limit(),
                    'relative gap tolerance': runner.relative_gap_tolerance(),
                    'hourly bids': runner.input_stats().number_of_hourly_bids(),
                    'block bids': runner.input_stats().number_of_block_bids(),
                    'flexible bids': runner.input_stats().number_of_flexible_bids(),
                    'valid':
                        False if runner.output().dam_solution() is None else runner.output().dam_solution().is_valid,
                    'total surplus':
                        -1 if runner.output().dam_solution() is None else
                        runner.output().dam_solution().total_surplus,
                    'solver status': runner.output().optimization_status().solver_status(),
                    'best bound': runner.output().optimization_status().best_bound(),
                    'relative gap': runner.output().optimization_status().relative_gap(),
                    'elapsed solver time': runner.output().optimization_stats().elapsed_time(),
                    'number of solutions': runner.output().optimization_stats().number_of_solutions(),
                    'number of nodes': runner.output().optimization_stats().number_of_nodes(),
                    'number of subproblems':
                        runner.output().optimization_stats().benders_decomposition_stats().number_of_subproblems_solved(),
                    'number of user cuts':
                        runner.output().optimization_stats().benders_decomposition_stats().number_of_user_cuts_added()
                }
            )
