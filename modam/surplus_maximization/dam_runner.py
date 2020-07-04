"""Implements runner class to solve the problem and return the output"""

import logging

from modam.surplus_maximization.dam_common import Solver, SolverParameters
from modam.surplus_maximization.dam_input import DamData, InputStats
from modam.surplus_maximization.dam_preprocessor import Preprocessor
from modam.surplus_maximization.dam_solver import DamSolverCplex, DamSolverGurobi, DamSolverScip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DamRunner(object):

    """Implements runner class for day-ahead market optimization"""

    def __init__(
            self, input_file_name, problem_type, solver, method, time_limit, relative_gap_tolerance, num_threads, 
            working_dir):
        self._input_file_name = input_file_name
        self._num_threads = num_threads
        self._problem_type = problem_type
        self._solver = solver
        self._method = method
        self._time_limit = time_limit
        self._relative_gap_tolerance = relative_gap_tolerance
        self._input_stats = None
        self._output = None
        self._working_dir = working_dir

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
        dam_data = DamData()
        dam_data.read_input(self._input_file_name)
        # create input data stats
        self._input_stats = InputStats(dam_data)
        # run pre-processor
        preprocessor = Preprocessor(dam_data, self._working_dir)
        dam_data = preprocessor.run()
        # solve problem
        params = SolverParameters(
            time_limit=self._time_limit, rel_gap=self._relative_gap_tolerance, num_threads=self._num_threads)
        if self._solver is Solver.Gurobi:
            dam_solver = DamSolverGurobi(self._problem_type, self._method, dam_data, params, self._working_dir)
        elif self._solver is Solver.Cplex:
            dam_solver = DamSolverCplex(self._problem_type, self._method, dam_data, params, self._working_dir)
        else:
            dam_solver = DamSolverScip(self._problem_type, self._method, dam_data, params, self._working_dir)
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
            self, input_file_names, problem_type, solver, method, time_limit, relative_gap_tolerance, num_threads, 
            working_dir):
        self._input_file_names = input_file_names
        self._problem_type = problem_type
        self._solver = solver
        self._method = method
        self._time_limit = time_limit
        self._relative_gap_tolerance = relative_gap_tolerance
        self._num_threads = num_threads
        self._runners = []
        self._working_dir = working_dir

    def runners(self):
        """Returns the runners created"""
        return self._runners

    def run(self):
        """Runs optimization for all configurations"""
        runners = self._runners
        for file_name in self._input_file_names:
            dam_runner = DamRunner(
                file_name, self._problem_type, self._solver, self._method, self._time_limit, 
                self._relative_gap_tolerance, self._num_threads, self._working_dir)
            dam_runner.run()
            runners.append(dam_runner)
