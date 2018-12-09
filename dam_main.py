# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:51:21 2018

@author: gokhanceyhan

Script to run dam clearing software
"""
import os
import logging
import sys
import csv

import dam_solver as ds
import dam_runner as dr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def single_run(filename, problem, solver, method):
    runner = dr.DamRunner(
        filename, problem_type=problem, solver=solver, method=method, time_limit=60, relative_gap_tolerance=1e-6)
    runner.run()
    if runner.output().optimization_stats().number_of_solutions() == 0:
        print('Failed to find a solution')
        sys.exit(-1)
    solution = runner.output().dam_solution()
    if solution.is_valid:
        print('Successfully found a solution!')
    else:
        print('Failed to find a valid solution')


def batch_run(input_folder_name):
    path = os.path.relpath(input_folder_name)
    input_file_names = [
        '/'.join([path, file]) for file in os.listdir(input_folder_name) if os.path.splitext(file)[1] == '.csv']
    problem_types = [ds.ProblemType.NoPab]
    solvers = [ds.Solver.Cplex]
    methods = [ds.SolutionApproach.Benders]
    time_limits = [60]
    relative_gap_tolerances = [1e-6]
    batch_runner = dr.BatchRunner(
        input_file_names, problem_types, solvers, methods, time_limits, relative_gap_tolerances)
    batch_runner.run()
    runners = batch_runner.runners()
    write_runners_to_file(runners)
    print('Runs have been completed!')


def write_runners_to_file(runners):

    with open('tests.csv', mode='w') as csv_file:
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


def usage():
    print('usage:   dam_main.py path run_mode problem solver method')
    print('path: relative path to the problem file(s)')
    print('run mode: {single, batch}')
    print('problem (required for run mode "single"): {NoPab, NoPrb}')
    print('solver (required for run mode "single"): {gurobi, cplex, scip}')
    print('method (required for run mode "single"): {primal-dual, benders, branch-bound(only scip)}')


if __name__ == "__main__":

    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)

    _path = sys.argv[1]
    _run_mode = sys.argv[2]
    if _run_mode == 'batch':
        batch_run(_path)
        sys.exit(-1)
    elif _run_mode == 'single':
        pass
    else:
        usage()
        sys.exit(-1)

    if len(sys.argv) < 5:
        usage()
        sys.exit(-1)

    _prob = None
    if sys.argv[3].lower() == 'nopab':
        _prob = ds.ProblemType.NoPab
    elif sys.argv[3].lower() == 'noprb':
        _prob = ds.ProblemType.NoPrb
    else:
        usage()
        sys.exit(-1)

    _solver = None
    if sys.argv[4].lower() == 'gurobi':
        _solver = ds.Solver.Gurobi
    elif sys.argv[4].lower() == 'cplex':
        _solver = ds.Solver.Cplex
    elif sys.argv[4].lower() == 'scip':
        _solver = ds.Solver.Scip
    else:
        usage()
        sys.exit(-1)

    _method = None
    if sys.argv[5].lower() == 'primal-dual':
        _method = ds.SolutionApproach.PrimalDual
    elif sys.argv[5].lower() == 'benders':
        _method = ds.SolutionApproach.Benders
    elif sys.argv[5].lower() == 'branch-and-bound':
        _method = ds.SolutionApproach.BranchAndBound
        if _solver is not ds.Solver.Scip:
            print('You can only use Scip')
            usage()
            sys.exit(-1)
    else:
        usage()
        sys.exit(-1)

    single_run(_path, _prob, _solver, _method)
