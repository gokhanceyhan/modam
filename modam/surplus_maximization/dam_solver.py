"""Implements dam solver"""

from enum import Enum
from abc import abstractmethod

import modam.surplus_maximization.dam_benders as db
from modam.surplus_maximization.dam_common import *
import modam.surplus_maximization.dam_primaldual as dpd
import modam.surplus_maximization.dam_branching as dbr
import modam.surplus_maximization.dam_utils as du


class DamSolver(object):

    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        self.prob_type = prob_type
        self.soln_app = soln_app
        self.dam_data = dam_data
        self.solver_params = solver_params

    @abstractmethod
    def solve(self):
        pass


class DamSolverCplex(DamSolver):

    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        DamSolver.__init__(self, prob_type, soln_app, dam_data, solver_params)

    def solve(self):
        if self.soln_app is SolutionApproach.Benders:
            dam_output = self._solve_benders_decomposition()
            return dam_output
        elif self.soln_app is SolutionApproach.PrimalDual:
            dam_output = self._solve_primal_dual_problem()
            return dam_output

    def _solve_primal_dual_problem(self):
        dam_pd = dpd.PrimalDualModel(self.prob_type, self.dam_data, 'e-smilp')
        prob_name = dam_pd.create_model()
        solver = dpd.PrimalDualCplexSolver(prob_name, self.solver_params)
        dam_output = solver.solve()
        return dam_output

    def _solve_benders_decomposition(self):
        dam_benders = db.BendersDecompositionCplex(self.prob_type, self.dam_data, self.solver_params)
        dam_output = dam_benders.solve()
        return dam_output


class DamSolverGurobi(DamSolver):

    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        DamSolver.__init__(self, prob_type, soln_app, dam_data, solver_params)

    def solve(self):
        if self.soln_app is SolutionApproach.Benders:
            dam_output = self._solve_benders_decomposition()
            return dam_output
        elif self.soln_app is SolutionApproach.PrimalDual:
            dam_output = self._solve_primal_dual_problem()
            return dam_output

    def _solve_primal_dual_problem(self):
        dam_pd = dpd.PrimalDualModel(self.prob_type, self.dam_data, 'e-smilp')
        prob_name = dam_pd.create_model()
        solver = dpd.PrimalDualGurobiSolver(prob_name, self.solver_params)
        dam_output = solver.solve()
        return dam_output

    def _solve_benders_decomposition(self):
        dam_benders = db.BendersDecompositionGurobi(self.prob_type, self.dam_data, self.solver_params)
        dam_output = dam_benders.solve()
        return dam_output


class DamSolverScip(DamSolver):

    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        DamSolver.__init__(self, prob_type, soln_app, dam_data, solver_params)

    def solve(self):
        if self.soln_app is SolutionApproach.Benders:
            dam_output = self._solve_benders_decomposition()
            return dam_output
        elif self.soln_app is SolutionApproach.PrimalDual:
            dam_output = self._solve_primal_dual_problem()
            return dam_output
        elif self.soln_app is SolutionApproach.BranchAndBound:
            dam_output = self._solve_branch_and_bound()
            return dam_output

    def _solve_primal_dual_problem(self):
        dam_pd = dpd.PrimalDualModel(self.prob_type, self.dam_data, 'e-smilp')
        prob_name = dam_pd.create_model()
        solver = dpd.PrimalDualScipSolver(prob_name, self.solver_params)
        dam_output = solver.solve()
        return dam_output

    def _solve_benders_decomposition(self):
        dam_benders = db.BendersDecompositionScip(self.prob_type, self.dam_data, self.solver_params)
        dam_output = dam_benders.solve()
        return dam_output

    def _solve_branch_and_bound(self):
        dam_branch_and_bound = dbr.BranchAndBoundScip(self.prob_type, self.dam_data, self.solver_params)
        dam_output = dam_branch_and_bound.solve()
        return dam_output
