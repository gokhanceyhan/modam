#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:51:21 2018

@author: gokhanceyhan

Script to run dam clearing software
"""

import dam_input as di
import dam_solver as ds
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(filename, problem, solver, method):
    # create dam input data
    logger.info('Start reading the input')
    dam_data = di.DamData()
    dam_data.read_input(filename)
    
    # create and print input data stats
    dam_data_stats = di.InputStats(dam_data)
    dam_data_stats.print_stats()

    # solve problem
    params = ds.SolverParameters(time_limit=10, rel_gap=1e-6)
    if solver is ds.Solver.Gurobi:
        dam_solver = ds.DamSolverGurobi(problem, method, dam_data, params)
    else:
        dam_solver = ds.DamSolverCplex(problem, method, dam_data, params)
    solution = dam_solver.solve()

    if solution is not None:
        print('Successfully found a solution!')
    else:
        print('Failed to find a solution')


def usage():
    print('Use python2.7 installation')
    print('usage:   dam_main.py filename problem solver method')
    print('problem: {NoPab, NoPrb}')
    print('solver: {gurobi, cplex, scip}')
    print('method: {primal-dual, benders}')


if __name__ == "__main__":
    if len(sys.argv) < 4:
        usage()
        sys.exit(-1)

    _filename = sys.argv[1]

    _prob = None
    if sys.argv[2].lower() == 'nopab':
        _prob = ds.ProblemType.NoPab
    elif sys.argv[2].lower() == 'noprb':
        _prob = ds.ProblemType.NoPrb
    else:
        usage()
        sys.exit(-1)

    _solver = None
    if sys.argv[3].lower() == 'gurobi':
        _solver = ds.Solver.Gurobi
    elif sys.argv[3].lower() == 'cplex':
        _solver = ds.Solver.Cplex
    elif sys.argv[3].lower == 'scip':
        _solver = ds.Solver.Scip
    else:
        usage()
        sys.exit(-1)

    _method = None
    if sys.argv[4].lower() == 'primal-dual':
        _method = ds.SolutionApproach.PrimalDual
    elif sys.argv[4].lower() == 'benders':
        _method = ds.SolutionApproach.Benders
    else:
        usage()
        sys.exit(-1)

    main(_filename, _prob, _solver, _method)
