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

from modam.surplus_maximization.dam_common import ProblemType, SolutionApproach, Solver
import modam.surplus_maximization.dam_runner as dr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def single_run(filename, problem, solver, method, time_limit=None, relative_gap_tolerance=None, num_threads=None):
    time_limit = time_limit or 600
    relative_gap_tolerance = relative_gap_tolerance or 1e-6
    num_threads = num_threads or 1
    runner = dr.DamRunner(
        filename, problem_type=problem, solver=solver, method=method, time_limit=time_limit,
        relative_gap_tolerance=relative_gap_tolerance, num_threads=num_threads)
    runner.run()
    if runner.output().optimization_stats().number_of_solutions() == 0:
        print('Failed to find a solution')
        sys.exit(-1)
    solution = runner.output().dam_solution()
    if solution.is_valid:
        print('Successfully found a solution!')
    else:
        print('Failed to find a valid solution')


def batch_run(
        input_folder_name, problem, solver, method, time_limit=None, relative_gap_tolerance=None, num_threads=None):
    path = os.path.relpath(input_folder_name)
    input_file_names = [
        '/'.join([path, file]) for file in os.listdir(input_folder_name) if os.path.splitext(file)[1] == '.csv']
    problem_types = problem if isinstance(problem, list) else [problem]
    solvers = solver if isinstance(solver, list) else [solver]
    methods = method if isinstance(method, list) else [method]
    time_limit = time_limit or 600
    relative_gap_tolerance = relative_gap_tolerance or 1e-6
    num_threads = num_threads or 1
    batch_runner = dr.BatchRunner(
        input_file_names, problem_types, solvers, methods, time_limit, relative_gap_tolerance, num_threads)
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
