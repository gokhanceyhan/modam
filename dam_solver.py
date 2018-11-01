from enum import Enum
from abc import abstractmethod
import dam_benders as db
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

    @abstractmethod
    def _get_solution(self, cont_model, bid_id_2_bbidvar, period_2_balance_con):
        pass


class DamSolverGurobi(DamSolver):
    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        DamSolver.__init__(self, prob_type, soln_app, dam_data, solver_params)

    def solve(self):
        if self.soln_app is SolutionApproach.Benders:
            dam_soln = self._solve_benders_decomposition()
            return dam_soln

    def _solve_primal_dual_problem(self):
        pass

    def _solve_benders_decomposition(self):
        dam_benders = db.BendersDecompositionGurobi(self.prob_type, self.dam_data, self.solver_params)
        dam_benders.solve()
        master = dam_benders.master_problem
        return self._get_solution(master.fixed, master.bid_id_2_bbidvar, master.period_2_balance_con)

    def _get_solution(self, cont_model, bid_id_2_bbidvar, period_2_balance_con):
        # fill solution
        dam_soln = DamSolution()
        dam_soln.total_surplus = cont_model.ObjVal
        y = cont_model.getAttr('X', bid_id_2_bbidvar)
        for bid_id, value in y.items():
            if abs(value - 0.0) <= cont_model.Params.IntFeasTol:
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        dam_soln.market_clearing_prices = cont_model.getAttr('Pi', period_2_balance_con)
        # verify solution
        dam_soln.verify(ProblemType.NoPab, self.dam_data.dam_bids)
        if dam_soln.is_valid:
            return dam_soln
        else:
            return None


class DamSolverCplex(DamSolver):
    def __init__(self, prob_type, soln_app, dam_data, solver_params):
        DamSolver.__init__(self, prob_type, soln_app, dam_data, solver_params)

    def solve(self):
        if self.soln_app is SolutionApproach.Benders:
            dam_soln = self._solve_benders_decomposition()
            return dam_soln

    def _solve_primal_dual_problem(self):
        pass

    def _solve_benders_decomposition(self):
        dam_benders = db.BendersDecompositionCplex(self.prob_type, self.dam_data, self.solver_params)
        dam_benders.solve()
        master = dam_benders.master_problem
        return self._get_solution(master.fixed, master.bid_id_2_bbidvar, master.period_2_balance_con)

    def _get_solution(self, cont_model, bid_id_2_bbidvar, period_2_balance_con):
        solution = cont_model.solution
        # fill dam solution object
        dam_soln = DamSolution()
        dam_soln.total_surplus = solution.get_objective_value()
        for bid_id, var in bid_id_2_bbidvar.values():
            value = solution.get_values(var)
            if abs(value - 0.0) <= cont_model.Params.IntFeasTol:
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        dam_soln.market_clearing_prices = solution.get_dual_values(list(period_2_balance_con.values()))
        # verify solution
        dam_soln.verify(ProblemType.NoPab, self.dam_data.dam_bids)
        if dam_soln.is_valid:
            return dam_soln
        else:
            return None


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
