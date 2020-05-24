"""Implements a solver factory for the multi-objective solver"""

from enum import Enum

from modam.multi_objective.nd_solver import NondominatedsetSolver


class SolverFactory:

    """Implements solver factory"""

    @staticmethod
    def _create_nondominated_set_solver(model_files, working_dir, lp_params_file=None, mip_params_file=None):
        """Creates a solver that generates the whole nondominated set"""
        return NondominatedsetSolver(
            model_files, working_dir, lp_params_file=lp_params_file, mip_params_file=mip_params_file)

    @staticmethod
    def create_solver(model_files, solver_type, working_dir, lp_params_file=None, mip_params_file=None):
        """Creates a solver of specified type"""
        if solver_type == SolverType.NONDOMINATED_SET_SOLVER:
            return SolverFactory._create_nondominated_set_solver(
                model_files, working_dir, lp_params_file=lp_params_file, mip_params_file=mip_params_file)
        raise NotImplementedError()


class SolverType:

    """Represents solver type"""

    GOAL_PROGRAMMING_SOLVER = "GP Solver"
    NONDOMINATED_SET_SOLVER = "ND Solver"

