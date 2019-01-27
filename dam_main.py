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


def usage():
    print('usage: dam_main.py path run_mode problem solver method')
    print('path: relative path to the problem file(s)')
    print('run mode: {single, batch}')
    print('problem: {Unrestricted, NoPab, NoPrb}')
    print('solver: {gurobi, cplex, scip}')
    print('method: {primal-dual, benders, branch-and-bound(only if the solver is scip)}')
    print('time limit in seconds: optional, default=600')
    print('relative mip gap tolerance: optional, default=1e-6')
    print('number of threads: optional, default=1 (cannot use scip with multiple threads)')


if __name__ == "__main__":

    if len(sys.argv) < 5:
        usage()
        sys.exit(-1)

    _prob = []
    if not sys.argv[3]:
        usage()
        sys.exit(-1)
    problems = sys.argv[3].split(',')
    for problem in problems:
        if problem.lower() == 'unrestricted':
            _prob.append(ds.ProblemType.Unrestricted)
        elif problem.lower() == 'nopab':
            _prob.append(ds.ProblemType.NoPab)
        elif problem.lower() == 'noprb':
            _prob.append(ds.ProblemType.NoPrb)

    _solver = []
    if not sys.argv[4]:
        usage()
        sys.exit(-1)
    solvers = sys.argv[4].split(',')
    for solver in solvers:
        if solver.lower() == 'gurobi':
            _solver.append(ds.Solver.Gurobi)
        elif solver.lower() == 'cplex':
            _solver.append(ds.Solver.Cplex)
        elif solver.lower() == 'scip':
            _solver.append(ds.Solver.Scip)

    _method = []
    if not sys.argv[5]:
        usage()
        sys.exit(-1)
    methods = sys.argv[5].split(',')
    for method in methods:
        if method.lower() == 'primal-dual':
            _method.append(ds.SolutionApproach.PrimalDual)
        elif method.lower() == 'benders':
            _method.append(ds.SolutionApproach.Benders)
        elif method.lower() == 'branch-and-bound':
            if not (len(_solver) == 1 and _solver[0] is ds.Solver.Scip):
                print('You can only use Scip with method branch-and-bound')
                usage()
                sys.exit(-1)
            _method.append(ds.SolutionApproach.BranchAndBound)

    _time_limit = None
    if len(sys.argv) > 5 and sys.argv[6] is not None:
        _time_limit = int(sys.argv[6])

    _relative_gap_tolerance = None
    if len(sys.argv) > 6 and sys.argv[7] is not None:
        _relative_gap_tolerance = float(sys.argv[7])

    _num_threads = None
    if len(sys.argv) > 7 and sys.argv[8] is not None:
        _num_threads = int(sys.argv[8])
        # if ds.Solver.Scip in _solver:
        #     print('Scip can only be run with single thread')
        #     usage()
        #     sys.exit(-1)

    _path = sys.argv[1]
    _run_mode = sys.argv[2]
    if _run_mode == 'batch':
        batch_run(
            _path, _prob, _solver, _method, time_limit=_time_limit, relative_gap_tolerance=_relative_gap_tolerance,
            num_threads=_num_threads)
        sys.exit(-1)
    elif _run_mode == 'single':
        single_run(
            _path, _prob[0], _solver[0], _method[0], time_limit=_time_limit,
            relative_gap_tolerance=_relative_gap_tolerance, num_threads=_num_threads)
        sys.exit(-1)
    else:
        usage()
        sys.exit(-1)



