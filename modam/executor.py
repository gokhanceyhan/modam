"""Implements executor functions"""

import argparse
from enum import Enum
import logging
import os

from modam.surplus_maximization.dam_common import ProblemType, SolutionApproach, Solver
from modam.surplus_maximization.dam_main import write_runners_to_file
from modam.surplus_maximization.dam_runner import BatchRunner, DamRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutorRunMode(Enum):

    """Represents executor run mode"""

    BATCH = "batch"
    SINGLE = "single"


class ExecutorApp:

    """Implements the command line application for the solver executor"""

    def _parse_args(self):
        """Parses and returns the arguments"""
        parser = argparse.ArgumentParser(description="day-ahead-market clearing problem solver executor app")
        parser.add_argument("-d", "--data-file-path", help="sets the path to the data file(s) (in .csv format)")
        parser.add_argument(
            "-g", "--mip-rel-gap", default=1e-6, 
            help="set the MIP relative gap tolerance in the surplus maximization solver")
        parser.add_argument("-m", "--multi-objective", action="store_true", help="use multi-objective solver")
        parser.add_argument(
            "--method", 
            choices=[
                SolutionApproach.Benders.value, SolutionApproach.BranchAndBound.value, 
                SolutionApproach.PrimalDual.value], 
            help="sets the method to use to solve the surplus maximiation problem")
        parser.add_argument(
            "--num-threads", default=1, help="set the number of threads in the surplus maximization MIP solver")
        parser.add_argument(
            "-p", "--problem-type", 
            choices=[ProblemType.NoPab.value, ProblemType.NoPrb.value, ProblemType.Unrestricted.value], 
            help="sets the problem type for the surplus maximization solver")
        parser.add_argument(
            "-r", "--run-mode", choices=[ExecutorRunMode.BATCH.value, ExecutorRunMode.SINGLE.value], 
            help="sets the run mode, e.g. single or batch run")
        parser.add_argument(
            "-s", "--solver", 
            choices=[Solver.Cplex.value, Solver.Gurobi.value, Solver.Scip.value], help="sets the solver to use")
        parser.add_argument(
            "-t", "--time-limit", default=600, 
            help="sets the time limit in seconds for the surplus maximization solver")
        parser.add_argument("-w", "--working-dir", help="sets the path to the working directory")
        return parser.parse_args()

    def run(self):
        """Runs the command line application"""
        args = self._parse_args()
        data_file_path = args.data_file_path
        run_mode = args.run_mode
        data_files = [os.path.join(data_file_path, f) for f in os.listdir(data_file_path) if f.endswith(".csv")] if \
            run_mode == ExecutorRunMode.BATCH.value else [data_file_path]
        multi_objective = args.multi_objective
        method = SolutionApproach(args.method)
        problem = ProblemType(args.problem_type)
        solver = Solver(args.solver)
        relative_gap_tolerance = float(args.mip_rel_gap)
        time_limit = int(args.time_limit)
        num_threads = int(args.num_threads)
        # a few argument value validations
        if method == SolutionApproach.BranchAndBound and solver != Solver.Scip:
            raise ValueError("the '%s' method can only be used with the '%s' solver" % (method.value, solver.value))
        if multi_objective and solver != Solver.Gurobi:
            raise ValueError("the multi-objective solver can only be used with the '%s' solver" % solver.value)
        if multi_objective:
            pass
        else:
            executor = SurplusMaximizationSolverExecutor(
                method=method, num_threads=num_threads, problem=problem, relative_gap_tolerance=relative_gap_tolerance, 
                solver=solver, time_limit_in_seconds=time_limit)
            executor.execute(data_file_path)


class MultiObjectiveSolverExecutor:

    """Implements multi-objective solver executor"""


class SurplusMaximizationSolverExecutor:

    """Implements surplus maximization solver executor"""

    def __init__(
            self, method=SolutionApproach.Benders, num_threads=1, problem=ProblemType.Unrestricted, 
            relative_gap_tolerance=1e-6, solver=Solver.Gurobi, time_limit_in_seconds=600):
        self._method = method
        self._num_threads = num_threads
        self._problem = problem
        self._relative_gap_tolerance = relative_gap_tolerance
        self._solver = solver
        self._time_limit_in_seconds = time_limit_in_seconds
    
    def batch_execute(self, file_names):
        """Executes the surplus maximization solver in batch for the given problem data files"""
        problem_types = [self._problem]
        solvers = [self._solver]
        methods = [self._method]
        batch_runner = BatchRunner(
            file_names, problem_types, solvers, methods, self._time_limit_in_seconds, self._relative_gap_tolerance, 
            self._num_threads)
        batch_runner.run()
        runners = batch_runner.runners()
        write_runners_to_file(runners)
        logger.info('Runs have been completed!')

    def execute(self, file_name):
        """Executes the surplus maximization solver for the given problem data file"""
        runner = DamRunner(
            file_name, problem_type=self._problem, solver=self._solver, method=self._method, 
            time_limit=self._time_limit_in_seconds, relative_gap_tolerance=self._relative_gap_tolerance, 
            num_threads=self._num_threads)
        runner.run()
        if runner.output().optimization_stats().number_of_solutions() == 0:
            logger.INFO('Failed to find a solution')
            return
        solution = runner.output().dam_solution()
        if solution.is_valid:
            logger.info('Successfully found a solution!')
        else:
            logger.info('Failed to find a valid solution')
