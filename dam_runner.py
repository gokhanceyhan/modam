"""Implements runner class to solve the problem and return the output"""

import logging

import dam_input as di
import dam_solver as ds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DamRunner(object):

    """Implements runner class for day-ahead market optimization"""

    def __init__(
            self, input_file_name, problem_type, solver, method, time_limit, relative_gap_tolerance, num_threads):
        self._input_file_name = input_file_name
        self._num_threads = num_threads
        self._problem_type = problem_type
        self._solver = solver
        self._method = method
        self._time_limit = time_limit
        self._relative_gap_tolerance = relative_gap_tolerance
        self._input_stats = None
        self._output = None

    def input_file_name(self):
        """Returns the input file name"""
        return self._input_file_name

    def problem_type(self):
        """Returns the problem type"""
        return self._problem_type

    def solver(self):
        """Returns the solver used to solve the problem"""
        return self._solver

    def method(self):
        """Returns the method used to solve the problem"""
        return self._method

    def num_threads(self):
        """Returns the number of threads to be used by the solver"""
        return self._num_threads

    def time_limit(self):
        """Returns the time limit set on the solver"""
        return self._time_limit

    def relative_gap_tolerance(self):
        """Returns the relative gap tolerance set on the solver"""
        return self._relative_gap_tolerance

    def input_stats(self):
        """Returns the input stats of the problem"""
        return self._input_stats

    def output(self):
        """Returns the output of the optimization"""
        return self._output

    def run(self):
        # create dam input data
        dam_data = di.DamData()
        dam_data.read_input(self._input_file_name)
        # create input data stats
        self._input_stats = di.InputStats(dam_data)
        # solve problem
        params = ds.SolverParameters(
            time_limit=self._time_limit, rel_gap=self._relative_gap_tolerance, num_threads=self._num_threads)
        if self._solver is ds.Solver.Gurobi:
            dam_solver = ds.DamSolverGurobi(self._problem_type, self._method, dam_data, params)
        elif self._solver is ds.Solver.Cplex:
            dam_solver = ds.DamSolverCplex(self._problem_type, self._method, dam_data, params)
        else:
            dam_solver = ds.DamSolverScip(self._problem_type, self._method, dam_data, params)
        # log run info
        logger.info('/'.join(
            [self._input_file_name, self._problem_type.value, self._solver.value, self._method.value,
             str(self._time_limit), str(self._relative_gap_tolerance)]))
        # solve
        output = dam_solver.solve()
        solution = output.dam_solution()
        if solution is not None:
            solution.verify(self._problem_type, dam_data.dam_bids)
        # create the output
        self._output = output


class BatchRunner(object):

    """Implements batch runner for day-ahead market optimization"""

    def __init__(
            self, input_file_names, problem_types, solvers, methods, time_limit, relative_gap_tolerance, num_threads):
        self._input_file_names = input_file_names
        self._problem_types = problem_types
        self._solvers = solvers
        self._methods = methods
        self._time_limit = time_limit
        self._relative_gap_tolerance = relative_gap_tolerance
        self._num_threads = num_threads
        self._runners = []

    def runners(self):
        """Returns the runners created"""
        return self._runners

    def run(self):
        """Runs optimization for all configurations"""
        runners = self._runners
        for file_name in self._input_file_names:
            for problem_type in self._problem_types:
                for solver in self._solvers:
                    for method in self._methods:
                        dam_runner = DamRunner(
                            file_name, problem_type, solver, method, self._time_limit, self._relative_gap_tolerance,
                            self._num_threads)
                        dam_runner.run()
                        runners.append(dam_runner)
