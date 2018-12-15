from enum import Enum
from abc import abstractmethod
import dam_benders as db
import dam_primaldual as dpd
import dam_branching as dbr
import dam_utils as du


class ProblemType(Enum):
    NoPab = 'NoPAB'
    NoPrb = 'NoPRB'
    Unrestricted = 'Unrestricted'


class SolutionApproach(Enum):
    PrimalDual = 'Primal-Dual'
    Benders = 'Benders Decomposition'
    BranchAndBound = 'Branch and Bound'


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


class DamSolution:
    def __init__(self):
        self.total_surplus = -1
        self.accepted_block_bids = []
        self.rejected_block_bids = []
        self.market_clearing_prices = []
        self.is_valid = False

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
        elif problem_type == ProblemType.Unrestricted:
            self.is_valid = True


class BendersDecompositionStats(object):

    """Implements statistics related to Benders decomposition algorithm"""

    def __init__(self, number_of_user_cuts_added=0, number_of_subproblems_solved=0):
        self._number_of_subproblems_solved = number_of_subproblems_solved
        self._number_of_user_cuts_added = number_of_user_cuts_added

    def number_of_subproblems_solved(self):
        """Returns the number of subproblems solved"""
        return self._number_of_subproblems_solved

    def number_of_user_cuts_added(self):
        """Returns the number of user cuts added"""
        return self._number_of_user_cuts_added


class OptimizationStats(object):

    """Implements optimization statistics"""

    def __init__(
            self, elapsed_time, number_of_nodes=0, number_of_solutions=0,
            benders_decomposition_stats=BendersDecompositionStats()):
        self._elapsed_time = elapsed_time
        self._number_of_nodes = number_of_nodes
        self._number_of_solutions = number_of_solutions
        self._benders_decomposition_stats = benders_decomposition_stats

    def benders_decomposition_stats(self):
        """Returns the benders decomposition stats"""
        return self._benders_decomposition_stats

    def elapsed_time(self):
        """Returns the elapsed time (wall-clock time) in seconds"""
        return self._elapsed_time

    def number_of_nodes(self):
        """Returns the number of nodes searched by the solver"""
        return self._number_of_nodes

    def number_of_solutions(self):
        """Returns the number of solutions found by the solver"""
        return self._number_of_solutions


class OptimizationStatus(object):

    """Implement optimization status"""

    def __init__(self, solver_status, relative_gap=None, best_bound=None):
        self._solver_status = solver_status
        self._relative_gap = relative_gap
        self._best_bound = best_bound

    def best_bound(self):
        """Returns the best bound"""
        return self._best_bound

    def relative_gap(self):
        """Returns the relative gap"""
        return self._relative_gap

    def solver_status(self):
        """Returns the solver status"""
        return self._solver_status


class DamSolverOutput(object):

    """Implements output of a solver used to solve day-ahead market clearing problem"""

    def __init__(self, dam_solution, optimization_stats, optimization_status):
        self._dam_solution = dam_solution
        self._optimization_stats = optimization_stats
        self._optimization_status = optimization_status

    def dam_solution(self):
        """Returns the day-ahead market solution"""
        return self._dam_solution

    def optimization_stats(self):
        """Returns the optimizations stats related to the dam solution"""
        return self._optimization_stats

    def optimization_status(self):
        """Returns the status of the dam solution with respect to optimality"""
        return self._optimization_status




