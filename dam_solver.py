from enum import Enum
from abc import abstractmethod
import dam_benders as db
import dam_primaldual as dpd
import dam_utils as du


class ProblemType(Enum):
    NoPab = 'NoPAB'
    NoPrb = 'NoPRB'


class SolutionApproach(Enum):
    PrimalDual = 'Primal-Dual'
    Benders = 'Benders Decomposition'


class Solver(Enum):
    Gurobi = 'Gurobi'
    Cplex = 'Cplex'
    Scip = 'Scip'


class SolverParameters:
    def __init__(self, time_limit=None, rel_gap=None):
        self.time_limit = time_limit
        self.rel_gap = rel_gap


class DamSolver(object):
    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        self.prob_type = prob_type
        self.soln_app = soln_app
        self.dam_data = dam_data
        self.solver_params = solver_params

    @abstractmethod
    def solve(self):
        pass


class DamSolverGurobi(DamSolver):
    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        DamSolver.__init__(self, prob_type, soln_app, dam_data, solver_params)

    def solve(self):
        if self.soln_app is SolutionApproach.Benders:
            dam_soln = self._solve_benders_decomposition()
            return dam_soln
        elif self.soln_app is SolutionApproach.PrimalDual:
            dam_soln = self._solve_primal_dual_problem()
            return dam_soln

    def _solve_primal_dual_problem(self):
        dam_pd = dpd.PrimalDualModel(self.prob_type, self.dam_data, 'e-smilp')
        prob_name = dam_pd.create_model()
        solver = dpd.PrimalDualGurobiSolver(prob_name, self.solver_params)
        solution = solver.solve()
        return solution

    def _solve_benders_decomposition(self):
        dam_benders = db.BendersDecompositionGurobi(self.prob_type, self.dam_data, self.solver_params)
        solution = dam_benders.solve()
        return solution


class DamSolverCplex(DamSolver):
    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        DamSolver.__init__(self, prob_type, soln_app, dam_data, solver_params)

    def solve(self):
        if self.soln_app is SolutionApproach.Benders:
            dam_soln = self._solve_benders_decomposition()
            return dam_soln
        elif self.soln_app is SolutionApproach.PrimalDual:
            dam_soln = self._solve_primal_dual_problem()
            return dam_soln

    def _solve_primal_dual_problem(self):
        dam_pd = dpd.PrimalDualModel(self.prob_type, self.dam_data, 'e-smilp')
        prob_name = dam_pd.create_model()
        solver = dpd.PrimalDualCplexSolver(prob_name, self.solver_params)
        solution = solver.solve()
        return solution

    def _solve_benders_decomposition(self):
        dam_benders = db.BendersDecompositionCplex(self.prob_type, self.dam_data, self.solver_params)
        solution = dam_benders.solve()
        return solution


class DamSolverScip(DamSolver):
    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        DamSolver.__init__(self, prob_type, soln_app, dam_data, solver_params)

    def solve(self):
        if self.soln_app is SolutionApproach.Benders:
            dam_soln = DamSolution()
            return dam_soln
        elif self.soln_app is SolutionApproach.PrimalDual:
            dam_soln = self._solve_primal_dual_problem()
            return dam_soln

    def _solve_primal_dual_problem(self):
        dam_pd = dpd.PrimalDualModel(self.prob_type, self.dam_data, 'e-smilp')
        prob_name = dam_pd.create_model()
        solver = dpd.PrimalDualScipSolver(prob_name, self.solver_params)
        solution = solver.solve()
        return solution

    def _solve_benders_decomposition(self):
        pass


class DamSolution:
    def __init__(self):
        self.total_surplus = None
        self.accepted_block_bids = []
        self.rejected_block_bids = []
        self.market_clearing_prices = []
        self.is_valid = None

    def _verify_no_pab(self, dam_bids):
        self.is_valid = True
        for bid_id in self.accepted_block_bids:
            bid = dam_bids.bid_id_2_block_bid[bid_id]
            if du.is_accepted_block_bid_pab(bid, self.market_clearing_prices):
                self.is_valid = False
                break

    def _verify_no_prb(self, dam_bids):
        self.is_valid = True
        for bid_id in self.rejected_block_bids:
            bid = dam_bids.bid_id_2_block_bid[bid_id]
            if du.is_rejected_block_bid_prb(bid, self.market_clearing_prices):
                self.is_valid = False
                break

    def verify(self, problem_type, dam_bids):
        if problem_type == ProblemType.NoPab:
            self._verify_no_pab(dam_bids)
        elif problem_type == ProblemType.NoPrb:
            self._verify_no_prb(dam_bids)
