from abc import abstractmethod
from collections import namedtuple
import os

import gurobipy as grb
import cplex as cpx
from cplex.callbacks import LazyConstraintCallback
import pyscipopt as scip

from modam.surplus_maximization.dam_common import BendersDecompositionStats, DamSolution, DamSolverOutput, \
    OptimizationStats, OptimizationStatus, ProblemType, SolutionApproach, Solver, SolverParameters
import modam.surplus_maximization.dam_constants as dc
from modam.surplus_maximization.dam_exceptions import UnsupportedProblemException
from modam.surplus_maximization.dam_input import BidType, ConstraintType, DamData
from modam.surplus_maximization.dam_post_problem import PostProblemGurobiModel
import modam.surplus_maximization.dam_utils as du


class BendersDecomposition(object):

    def __init__(self, prob_type, dam_data, solver_params, working_dir):
        self.prob_type = prob_type
        self.dam_data = dam_data
        self.solver_params = solver_params
        self.master_problem = None
        self.sub_problem = None
        self._working_dir = working_dir

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def _solve_no_pab(self):
        pass

    @abstractmethod
    def _solve_no_prb(self):
        pass

    @abstractmethod
    def _get_best_solution(self):
        pass

    @abstractmethod
    def _get_solver_output(self, *args):
        pass


class MasterProblem(object):

    def __init__(self, dam_data, working_dir):
        self.dam_data = dam_data
        self.model = None
        self.fixed = None
        self.relaxed = None
        self.bid_id_2_hbidvars = {}
        self.bid_id_2_bbidvar = {}
        self.period_2_balance_con = {}
        self._working_dir = working_dir

    @abstractmethod
    def _create_hbidvars(self):
        pass

    @abstractmethod
    def _create_bbidvars(self):
        pass

    @abstractmethod
    def _create_obj_function(self):
        pass

    @abstractmethod
    def _create_balance_constraints(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def write_model(self):
        pass

    @abstractmethod
    def set_params(self, solver_params):
        pass

    @abstractmethod
    def solve_model(self):
        pass

    @abstractmethod
    def solve_model_with_callback(self, *args):
        pass

    @abstractmethod
    def solve_fixed_model(self):
        pass

    @abstractmethod
    def solve_relaxed_model(self):
        pass


class SubProblem(object):

    def __init__(self, working_dir, *args):
        self._working_dir = working_dir

    @abstractmethod
    def reset_block_bid_bounds(self, *args):
        pass

    @abstractmethod
    def restrict_rejected_block_bids(self, rejected_block_bids):
        pass

    @abstractmethod
    def restrict_accepted_block_bids(self, accepted_block_bids):
        pass

    @abstractmethod
    def write_model(self):
        pass

    @abstractmethod
    def solve_model(self):
        pass


class BendersDecompositionGurobi(BendersDecomposition):

    def solve(self):
        if self.prob_type is ProblemType.NoPab:
            return self._solve_no_pab()
        elif self.prob_type is ProblemType.NoPrb:
            return self._solve_no_prb()
        elif self.prob_type is ProblemType.Unrestricted:
            return self._solve_unrestricted_problem()
        else:
            raise UnsupportedProblemException(
                "Problem type %s is not supported in method Benders Decomposition of solver Gurobi" % self.prob_type)

    def _solve_no_pab(self):
        # create master problem
        self.master_problem = MasterProblemGurobi(self.dam_data, self._working_dir)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)

        # create sub problem
        sub_prob = MasterProblemGurobi(self.dam_data, self._working_dir)
        sub_prob.create_model()
        self.sub_problem = SubProblemGurobi(sub_prob.model.relax(), self._working_dir)
        sub_prob = self.sub_problem

        # pass data into callback
        master_prob.model._dam_data = self.dam_data
        master_prob.model._bid_id_2_bbidvar = master_prob.bid_id_2_bbidvar
        master_prob.model._sp = sub_prob
        master_prob.model._prob = ProblemType.NoPab
        master_prob.model._num_of_subproblems = 0
        master_prob.model._num_of_user_cuts = 0

        # run benders decomposition
        callback = CallbackGurobi()
        master_prob.solve_model_with_callback(callback.dam_callback)
        return self._get_solver_output()

    def _solve_no_prb(self):
        # create master problem
        self.master_problem = MasterProblemGurobi(self.dam_data, self._working_dir)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)

        # create sub problem
        sub_prob = MasterProblemGurobi(self.dam_data, self._working_dir)
        sub_prob.create_model()
        self.sub_problem = SubProblemGurobi(sub_prob.model.relax(), self._working_dir)
        sub_prob = self.sub_problem

        # pass data into callback
        master_prob.model._dam_data = self.dam_data
        master_prob.model._bid_id_2_bbidvar = master_prob.bid_id_2_bbidvar
        master_prob.model._sp = sub_prob
        master_prob.model._prob = ProblemType.NoPrb
        master_prob.model._num_of_subproblems = 0
        master_prob.model._num_of_user_cuts = 0

        # run benders decomposition
        callback = CallbackGurobi()
        master_prob.solve_model_with_callback(callback.dam_callback)
        return self._get_solver_output()

    def _solve_unrestricted_problem(self):
        # create master problem
        self.master_problem = MasterProblemGurobi(self.dam_data, self._working_dir)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)
        master_prob.model._num_of_subproblems = 0
        master_prob.model._num_of_user_cuts = 0
        master_prob.solve_model()
        return self._get_solver_output()

    def _get_best_solution(self, use_post_processing_problem=False):
        # fill solution
        mp = self.master_problem
        # the block bids in 'dam_bids' include both original block bids and the block bids generated from flexible bids
        bid_id_2_block_bid = self.dam_data.dam_bids.bid_id_2_block_bid
        bid_id_2_flexible_bid = self.dam_data.dam_original_bids.bid_id_2_flexible_bid
        exclusive_group_id_2_block_bid_ids = self.dam_data.exclusive_group_id_2_block_bid_ids
        dam_soln = DamSolution()
        dam_soln.total_surplus = mp.fixed.ObjVal
        y = mp.fixed.getAttr('X', mp.bid_id_2_bbidvar)
        block_bid_id_2_value = {}
        for bid_id, value in y.items():
            bid = bid_id_2_block_bid[bid_id]
            if abs(value - 0.0) <= mp.fixed.Params.IntFeasTol:
                block_bid_id_2_value[bid_id] = 0
                if not bid.from_flexible:
                    dam_soln.rejected_block_bids.append(bid_id)
            else:
                block_bid_id_2_value[bid_id] = 1
                if bid.from_flexible:
                    dam_soln.accepted_block_bids_from_flexible_bids.append(bid_id)
                    flexible_bid_id = bid.exclusive_group_id
                    if flexible_bid_id not in dam_soln.accepted_flexible_bids:
                        dam_soln.accepted_flexible_bids.append(flexible_bid_id)
                elif bid.exclusive_group_id is not None and \
                        bid.exclusive_group_id not in dam_soln.accepted_mutually_exclusive_block_bid_group_ids:
                    dam_soln.accepted_mutually_exclusive_block_bid_group_ids.append(bid.exclusive_group_id)
                else:
                    dam_soln.accepted_block_bids.append(bid_id)
        dam_soln.rejected_flexible_bids = [
            bid_id for bid_id in bid_id_2_flexible_bid if bid_id not in dam_soln.accepted_flexible_bids]
        dam_soln.rejected_mutually_exclusive_block_bid_group_ids = [
            group_id for group_id in exclusive_group_id_2_block_bid_ids if group_id not in 
            dam_soln.accepted_mutually_exclusive_block_bid_group_ids]
        post_problem_result = None
        if use_post_processing_problem:
            post_problem = PostProblemGurobiModel(self.dam_data, self._working_dir)
            post_problem_result = post_problem.solve(block_bid_id_2_value=block_bid_id_2_value)
        if post_problem_result:
            dam_soln.market_clearing_prices = post_problem_result.prices()
        else:
            dam_soln.market_clearing_prices = mp.fixed.getAttr('Pi', mp.period_2_balance_con.values())
        dam_soln = du.generate_market_result_statistics(self.dam_data, dam_soln)
        return dam_soln

    def _get_solver_output(self):
        # collect optimization stats
        model = self.master_problem.model
        elapsed_time = model.Runtime
        number_of_solutions = model.SolCount
        number_of_nodes = model.NodeCount
        number_of_subproblems_solved = model._num_of_subproblems
        number_of_user_cuts_added = model._num_of_user_cuts
        benders_stats = BendersDecompositionStats(
            number_of_user_cuts_added=number_of_user_cuts_added,
            number_of_subproblems_solved=number_of_subproblems_solved)
        optimization_stats = OptimizationStats(
            elapsed_time, number_of_nodes, number_of_solutions, benders_decomposition_stats=benders_stats)
        # collect optimization status
        status = model.Status
        best_bound = model.ObjBound
        mip_relative_gap = model.MIPGap
        optimization_status = OptimizationStatus(status, mip_relative_gap, best_bound)
        # best solution query
        if number_of_solutions >= 1:
            self.master_problem.solve_fixed_model()
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        return DamSolverOutput(best_solution, optimization_stats, optimization_status)


class MasterProblemGurobi(MasterProblem):

    def __init__(self, dam_data, working_dir):
        MasterProblem.__init__(self, dam_data, working_dir)
        self.model = grb.Model('master')

    def _create_hbidvars(self):
        # step hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_simple_bid.keys():
                step_id_2_sbidvar[step_id] = self.model.addVar(
                    vtype=grb.GRB.CONTINUOUS, name='x_' + str(bid_id) + '_' + str(step_id), lb=0, ub=1)
            self.bid_id_2_hbidvars[bid_id] = step_id_2_sbidvar
        # piecewise hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_interpolated_bid.keys():
                step_id_2_sbidvar[step_id] = self.model.addVar(
                    vtype=grb.GRB.CONTINUOUS, name='x_' + str(bid_id) + '_' + str(step_id), lb=0, ub=1)
            self.bid_id_2_hbidvars[bid_id] = step_id_2_sbidvar

    def _create_bbidvars(self):
        for bid_id in self.dam_data.dam_bids.bid_id_2_block_bid.keys():
            self.bid_id_2_bbidvar[bid_id] = self.model.addVar(vtype=grb.GRB.BINARY, name='y_' + str(bid_id))

    def _create_obj_function(self):
        lin_expr = grb.LinExpr(0.0)
        quad_expr = grb.QuadExpr(0.0)
        # set coefficients for simple bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                lin_expr.add(self.bid_id_2_hbidvars[bid_id][step_id], simple_bid.p * simple_bid.q)
        # set coefficients for interpolated bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid.items():
            for step_id, interpolated_bid in hourly_bid.step_id_2_interpolated_bid.items():
                svar = self.bid_id_2_hbidvars[bid_id][step_id]
                lin_expr.add(svar, interpolated_bid.p_start * interpolated_bid.q)
                quad_expr.add(
                    svar * svar, 0.5 * (interpolated_bid.p_end - interpolated_bid.p_start) * interpolated_bid.q)
        # set coefficients for block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            lin_expr.add(self.bid_id_2_bbidvar[bid_id], block_bid.price * block_bid.total_quantity)
        self.model.setObjective(lin_expr + quad_expr, grb.GRB.MAXIMIZE)

    def _create_balance_constraints(self):
        period_2_expr = {}
        for period in range(1, DamData.NUM_PERIODS + 1, 1):
            expr = grb.LinExpr(0.0)
            period_2_expr[period] = expr
        # step hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            expr = period_2_expr[hourly_bid.period]
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                expr.add(self.bid_id_2_hbidvars[bid_id][step_id], simple_bid.q)
        # piecewise hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid.items():
            expr = period_2_expr[hourly_bid.period]
            for step_id, interpolated_bid in hourly_bid.step_id_2_interpolated_bid.items():
                expr.add(self.bid_id_2_hbidvars[bid_id][step_id], interpolated_bid.q)
        # block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            for t in range(block_bid.period, block_bid.period + block_bid.num_period, 1):
                expr = period_2_expr[t]
                expr.add(self.bid_id_2_bbidvar[bid_id], block_bid.quantity(t))
        for period, expr in period_2_expr.items():
            constraint = self.model.addConstr(expr, grb.GRB.EQUAL, 0.0, 'balance_' + str(period))
            self.period_2_balance_con[period] = constraint

    def _create_block_bid_constraints(self):
        """Creates the constraints for the linked block bids, mutually exclusive block bids and flexible bids"""
        constraints = self.dam_data.block_bid_constraints_matrix
        rhs = self.dam_data.block_bid_constraints_rhs
        vars_ = [self.bid_id_2_bbidvar[bid_id] for bid_id in self.dam_data.block_bid_constraints_bid_ids]
        types = self.dam_data.block_bid_constraints_types
        constraint_type_2_constraint_sense = {
            ConstraintType.EQUAL_TO: grb.GRB.EQUAL,
            ConstraintType.GREATER_THAN_EQUAL_TO: grb.GRB.GREATER_EQUAL,
            ConstraintType.LESS_THAN_EQUAL_TO: grb.GRB.LESS_EQUAL}
        for index, (constraint, rhs_, type_) in enumerate(zip(constraints, rhs, types)):
            sense = constraint_type_2_constraint_sense[type_]
            name = f"block_bid_constraints_{index}"
            self.model.addConstr(grb.quicksum([c * var for c, var in zip(constraint, vars_)]), sense, rhs_, name)

    def _create_cuts_for_identical_bids(self):
        """Sets a complete ordering for the acceptance of identical bid sets
        
        NOTE: Implemented only for block bids"""
        block_bid_lists = self.dam_data.dam_bids.bid_type_2_identical_bid_lists.get(BidType.BLOCK, [])
        model = self.model
        for block_bids in block_bid_lists:
            for index, block_bid in enumerate(block_bids):
                if index == len(block_bids) - 1:
                    continue
                bid_id = block_bid.bid_id
                bid_id_ = block_bids[index + 1].bid_id
                y = self.bid_id_2_bbidvar[bid_id]
                y_ = self.bid_id_2_bbidvar[bid_id_]
                model.addConstr(y_ - y, grb.GRB.LESS_EQUAL, 0, "identical_bid_ordering_" + bid_id + "_" + bid_id_)

    def create_model(self):
        # create decision variables
        self._create_hbidvars()
        self._create_bbidvars()
        # create objective function
        self._create_obj_function()
        # create constraint set
        self._create_balance_constraints()
        self._create_block_bid_constraints()
        # create identical bid cuts
        self._create_cuts_for_identical_bids()
        self.model.update()

    def write_model(self):
        # write model
        self.model.write(os.path.join(self._working_dir, 'master.lp'))

    def set_params(self, solver_params):
        self.model.Params.LogToConsole = 0
        self.model.Params.MIPGap = solver_params.rel_gap
        self.model.Params.TimeLimit = solver_params.time_limit
        self.model.Params.Threads = solver_params.num_threads

    def solve_model(self):
        # solve model
        self.model.optimize()

    def solve_model_with_callback(self, callback):
        # self.model.Params.Heuristics = 0
        self.model.Params.LazyConstraints = 1
        self.model.optimize(callback)

    def solve_fixed_model(self):
        # solve restricted lp model
        self.fixed = self.model.fixed()
        self.fixed.optimize()

    def solve_relaxed_model(self):
        self.relaxed = self.model.copy().relax()
        self.relaxed.optimize()


class SubProblemGurobi(SubProblem):

    def __init__(self, model, working_dir):
        SubProblem.__init__(self, working_dir)
        self.model = model
        self.objval = None
        self.set_params()

    def reset_block_bid_bounds(self, bid_id_2_bbidvar):
        for bid_id in bid_id_2_bbidvar.keys():
            var = self.model.getVarByName('y_' + bid_id)
            var.lb = 0.0
            var.ub = 1.0

    def restrict_rejected_block_bids(self, rejected_block_bids):
        # restrict the model
        for bid_id in rejected_block_bids:
            var = self.model.getVarByName('y_' + bid_id)
            var.ub = 0.0

    def restrict_accepted_block_bids(self, accepted_block_bids):
        # restrict the model
        for bid_id in accepted_block_bids:
            var = self.model.getVarByName('y_' + bid_id)
            var.lb = 1.0

    def set_params(self):
        # set parameters
        self.model.Params.OutputFlag = 0

    def write_model(self):
        # write model
        self.model.write(os.path.join(self._working_dir, 'sub.lp'))

    def solve_model(self):
        # solve model
        self.model.optimize()
        self.objval = self.model.ObjVal


class CallbackGurobi:

    def __init__(self):
        pass

    @staticmethod
    def dam_callback(model, where):
        if where == grb.GRB.Callback.MIPSOL:
            # query node obj value
            node_obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
            # query node solution
            accepted_block_bids = []
            rejected_block_bids = []
            y = model.cbGetSolution(model._bid_id_2_bbidvar)
            for bid_id, value in y.items():
                if abs(value - 0.0) <= model.Params.IntFeasTol:
                    rejected_block_bids.append(bid_id)
                else:
                    accepted_block_bids.append(bid_id)
            # update sub problem
            model._sp.reset_block_bid_bounds(model._bid_id_2_bbidvar)
            if model._prob is ProblemType.NoPab:
                model._sp.restrict_rejected_block_bids(rejected_block_bids)
            elif model._prob is ProblemType.NoPrb:
                model._sp.restrict_accepted_block_bids(accepted_block_bids)
            # solve sub problem
            model._sp.solve_model()
            model._num_of_subproblems += 1
            # assess if the current node solution is valid
            sub_problem_obj = model._sp.objval
            if sub_problem_obj > node_obj + dc.OBJ_COMP_TOL:
                # add lazy constraint to cut this solution
                CallbackGurobi._generate_lazy_cuts(model, accepted_block_bids, rejected_block_bids)

    @staticmethod
    def _generate_lazy_cuts(model, accepted_block_bids, rejected_block_bids):
        CallbackGurobi._generate_gcuts(model, accepted_block_bids, rejected_block_bids)

    @staticmethod
    def _generate_combinatorial_cut_martin(model, accepted_block_bids, rejected_block_bids, bid_id_2_bbidvar):
        expr = grb.LinExpr(0.0)
        rhs = 1
        for bid_id in accepted_block_bids:
            bid_var = bid_id_2_bbidvar[bid_id]
            expr.add(bid_var, -1)
            rhs -= 1
        for bid_id in rejected_block_bids:
            bid_var = bid_id_2_bbidvar[bid_id]
            expr.add(bid_var, 1)
        model.cbLazy(expr >= rhs)
        model._num_of_user_cuts += 1

    @staticmethod
    def _generate_gcuts(model, accepted_block_bids, rejected_block_bids):
        bid_id_2_block_bid = model._dam_data.dam_bids.bid_id_2_block_bid
        block_bid_id_2_child_block_bids = model._dam_data.block_bid_id_2_child_block_bids
        exclusive_group_id_2_block_bid_ids = model._dam_data.exclusive_group_id_2_block_bid_ids
        bid_id_2_bbidvar = model._bid_id_2_bbidvar
        market_clearing_prices = CallbackGurobi._find_market_clearing_prices(
            model, accepted_block_bids, rejected_block_bids)
        pabs = du.find_pabs(
            market_clearing_prices, accepted_block_bids, bid_id_2_block_bid, block_bid_id_2_child_block_bids) \
            if model._prob is ProblemType.NoPab else []
        prbs = du.find_prbs(
            market_clearing_prices, rejected_block_bids, bid_id_2_block_bid, accepted_block_bids, 
            exclusive_group_id_2_block_bid_ids) if model._prob is ProblemType.NoPrb else []
        for pab in pabs:
            variables, coefficients, rhs = du.create_gcut_for_pab(
                pab, accepted_block_bids, rejected_block_bids, bid_id_2_block_bid, bid_id_2_bbidvar)
            expr = grb.LinExpr(0.0)
            expr.addTerms(coefficients, variables)
            model.cbLazy(expr >= rhs)
            model._num_of_user_cuts += 1
        for prb in prbs:
            variables, coefficients, rhs = du.create_gcut_for_prb(
                prb, accepted_block_bids, rejected_block_bids, bid_id_2_block_bid, bid_id_2_bbidvar, 
                exclusive_group_id_2_block_bid_ids)
            expr = grb.LinExpr(0.0)
            expr.addTerms(coefficients, variables)
            model.cbLazy(expr >= rhs)
            model._num_of_user_cuts += 1

    @staticmethod
    def _find_market_clearing_prices(model, accepted_block_bids, rejected_block_bids):
        # solve sub-problem again to obtain dual values
        # restrict accepted and rejected block bids
        sp = model._sp
        dam_data = model._dam_data
        sp.restrict_accepted_block_bids(accepted_block_bids)
        sp.restrict_rejected_block_bids(rejected_block_bids)
        sp.solve_model()
        # TODO: replace 'balance_' with a defined constant
        balance_constraints = [
            sp.model.getConstrByName('balance_' + str(period)) for period in 
            range(1, DamData.NUM_PERIODS + 1, 1)]
        market_clearing_prices = sp.model.getAttr('Pi', balance_constraints)
        return market_clearing_prices


class BendersDecompositionCplex(BendersDecomposition):

    """Note: Block bid variable indices are communicated across the class instead of bid_ids and variables."""

    def solve(self):
        if self.prob_type is ProblemType.NoPab:
            return self._solve_no_pab()
        elif self.prob_type is ProblemType.NoPrb:
            return self._solve_no_prb()
        elif self.prob_type is ProblemType.Unrestricted:
            return self._solve_unrestricted_problem()
        else:
            raise UnsupportedProblemException(
                "Problem type %s is not supported in method Benders Decomposition of solver Cplex" % self.prob_type)

    def _solve_no_pab(self):
        # create master problem
        self.master_problem = MasterProblemCplex(self.dam_data, self._working_dir)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)

        # create sub problem
        self.sub_problem = SubProblemCplex(master_prob, self._working_dir)
        sub_prob = self.sub_problem

        # run benders decomposition
        start_time = master_prob.model.get_time()
        master_prob.solve_model_with_callback(sub_prob, ProblemType.NoPab)
        end_time = master_prob.model.get_time()
        elapsed_time = end_time - start_time
        return self._get_solver_output(elapsed_time)

    def _solve_no_prb(self):
        # create master problem
        self.master_problem = MasterProblemCplex(self.dam_data, self._working_dir)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)

        # create sub problem
        self.sub_problem = SubProblemCplex(master_prob, self._working_dir)
        sub_prob = self.sub_problem

        # run benders decomposition
        start_time = master_prob.model.get_time()
        master_prob.solve_model_with_callback(sub_prob, ProblemType.NoPrb)
        end_time = master_prob.model.get_time()
        elapsed_time = end_time - start_time
        return self._get_solver_output(elapsed_time)

    def _solve_unrestricted_problem(self):
        # create master problem
        self.master_problem = MasterProblemCplex(self.dam_data, self._working_dir)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)
        # run benders decomposition
        start_time = master_prob.model.get_time()
        master_prob.solve_model()
        end_time = master_prob.model.get_time()
        elapsed_time = end_time - start_time
        return self._get_solver_output(elapsed_time)

    def _get_best_solution(self):
        mp = self.master_problem
        solution = mp.fixed.solution
        # solution.write('solution.sol')
        # fill dam solution object
        dam_soln = DamSolution()
        dam_soln.total_surplus = solution.get_objective_value()
        for bid_id, var in mp.bid_id_2_bbidvar.items():
            value = solution.get_values(var)
            if abs(value - 0.0) <= mp.fixed.parameters.mip.tolerances.integrality.get():
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        dam_soln.market_clearing_prices = solution.get_dual_values(mp.period_2_balance_con)
        dam_soln = du.generate_market_result_statistics(self.dam_data.dam_bids, dam_soln)
        return dam_soln

    def _get_solver_output(self, elapsed_time):
        master_problem = self.master_problem
        model = master_problem.model
        solution = model.solution
        # collect optimization stats
        elapsed_time = elapsed_time
        number_of_solutions = solution.pool.get_num()
        number_of_nodes = solution.progress.get_num_nodes_processed()
        number_of_subproblems_solved = master_problem.callback_instance._times_called if \
            master_problem.callback_instance else 0
        number_of_user_cuts_added = master_problem.callback_instance._cuts_added if \
            master_problem.callback_instance else 0
        benders_stats = BendersDecompositionStats(
            number_of_user_cuts_added=number_of_user_cuts_added,
            number_of_subproblems_solved=number_of_subproblems_solved)
        optimization_stats = OptimizationStats(
            elapsed_time, number_of_nodes, number_of_solutions, benders_decomposition_stats=benders_stats)
        # collect optimization status
        status = solution.get_status()
        best_bound = solution.MIP.get_best_objective()
        mip_relative_gap = solution.MIP.get_mip_relative_gap() if number_of_solutions >= 1 else -1
        optimization_status = OptimizationStatus(status, mip_relative_gap, best_bound)
        # best solution query
        if number_of_solutions >= 1:
            master_problem.solve_fixed_model()
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        return DamSolverOutput(best_solution, optimization_stats, optimization_status)


class MasterProblemCplex(MasterProblem):

    def __init__(self, dam_data, working_dir):
        MasterProblem.__init__(self, dam_data, working_dir)
        self.model = cpx.Cplex()
        self.name_2_ind = None
        self.callback_instance = None

    def _create_hbidvars(self):
        model = self.model
        # step hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_simple_bid:
                var_name = 'x_' + str(bid_id) + '_' + str(step_id)
                model.variables.add(lb=[0.0], ub=[1.0], types=['C'], names=[var_name])
                step_id_2_sbidvar[step_id] = var_name
            self.bid_id_2_hbidvars[bid_id] = step_id_2_sbidvar
        # piecewise hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_interpolated_bid:
                var_name = 'x_' + str(bid_id) + '_' + str(step_id)
                model.variables.add(lb=[0.0], ub=[1.0], types=['C'], names=[var_name])
                step_id_2_sbidvar[step_id] = var_name
            self.bid_id_2_hbidvars[bid_id] = step_id_2_sbidvar

    def _create_bbidvars(self):
        for bid_id, bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            var_name = 'y_' + str(bid_id)
            self.model.variables.add(lb=[0.0], ub=[1.0], types=['B'], names=[var_name])
            self.bid_id_2_bbidvar[bid_id] = var_name

    def _create_obj_function(self):
        obj = self.model.objective
        # set coefficients for simple bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                obj.set_linear(self.bid_id_2_hbidvars[bid_id][step_id], simple_bid.p * simple_bid.q)
        # set coefficients for interpolated bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid.items():
            for step_id, interpolated_bid in hourly_bid.step_id_2_interpolated_bid.items():
                svar = self.bid_id_2_hbidvars[bid_id][step_id]
                obj.set_linear(svar, interpolated_bid.p_start * interpolated_bid.q)
                # the multiplier 0.5 is dropped since it is already accounted for in the quadratic coefficients in Cplex
                # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/refpythoncplex/html/
                # cplex._internal._subinterfaces.ObjectiveInterface-class.html#set_quadratic_coefficients
                obj.set_quadratic_coefficients(
                    svar, svar, (interpolated_bid.p_end - interpolated_bid.p_start) * interpolated_bid.q)
        # set coefficients for block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            obj.set_linear(self.bid_id_2_bbidvar[bid_id], block_bid.price * block_bid.total_quantity)
        # set obj sense
        obj.set_sense(self.model.objective.sense.maximize)

    def _create_balance_constraints(self):
        inds = [[] for i in range(DamData.NUM_PERIODS)]
        vals = [[] for i in range(DamData.NUM_PERIODS)]
        # step hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            period = hourly_bid.period
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                var_name = self.bid_id_2_hbidvars[bid_id][step_id]
                inds[period - 1].append(self.model.variables.get_indices(var_name))
                vals[period - 1].append(simple_bid.q)
        # piecewise hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid.items():
            period = hourly_bid.period
            for step_id, interpolated_bid in hourly_bid.step_id_2_interpolated_bid.items():
                var_name = self.bid_id_2_hbidvars[bid_id][step_id]
                inds[period - 1].append(self.model.variables.get_indices(var_name))
                vals[period - 1].append(interpolated_bid.q)
        # block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            var_name = self.bid_id_2_bbidvar[bid_id]
            ind = self.model.variables.get_indices(var_name)
            for t in range(block_bid.period, block_bid.period + block_bid.num_period, 1):
                inds[t - 1].append(ind)
                vals[t - 1].append(block_bid.quantity(t))

        con_expr = [cpx.SparsePair(inds[t - 1], vals[t - 1]) for t in range(1, DamData.NUM_PERIODS + 1, 1)]
        senses = ['E'] * DamData.NUM_PERIODS
        rhs = [0.0] * DamData.NUM_PERIODS
        con_names = ['balance_' + str(period) for period in range(1, DamData.NUM_PERIODS + 1, 1)]
        self.model.linear_constraints.add(lin_expr=con_expr, senses=senses, rhs=rhs, names=con_names)
        self.period_2_balance_con = con_names

    def _create_cuts_for_identical_bids(self):
        """Sets a complete ordering for the acceptance of identical bid sets
        
        NOTE: Implemented only for block bids"""
        block_bid_lists = self.dam_data.dam_bids.bid_type_2_identical_bid_lists.get(BidType.BLOCK, [])
        model = self.model
        for block_bids in block_bid_lists:
            for index, block_bid in enumerate(block_bids):
                if index == len(block_bids) - 1:
                    continue
                bid_id = block_bid.bid_id
                bid_id_ = block_bids[index + 1].bid_id
                y = model.variables.get_indices(self.bid_id_2_bbidvar[bid_id])
                y_ = model.variables.get_indices(self.bid_id_2_bbidvar[bid_id_])
                expr = cpx.SparsePair(ind=[y, y_], val=[-1, 1])
                name = "identical_bid_ordering_" + bid_id + "_" + bid_id_
                model.linear_constraints.add(lin_expr=[expr], senses=["L"], rhs=[0], names=[name])

    def create_model(self):
        # create decision variables
        self._create_hbidvars()
        self._create_bbidvars()
        # create objective function
        self._create_obj_function()
        # create constraint set
        self._create_balance_constraints()
        # create identical bid constraints
        self._create_cuts_for_identical_bids()
        # create name_2_ind dictionary
        self.name_2_ind = {n: j for j, n in enumerate(self.model.variables.get_names())}

    def write_model(self):
        self.model.write(os.path.join(self._working_dir, "master.lp"))

    def set_params(self, solver_params):
        self.model.parameters.mip.tolerances.mipgap.set(solver_params.rel_gap)
        self.model.parameters.timelimit.set(solver_params.time_limit)
        self.model.parameters.threads.set(solver_params.num_threads)
        log_file = os.path.join(self._working_dir, "cplex.log")
        self.model.set_log_stream(log_file)
        self.model.set_results_stream(log_file)
        self.model.set_warning_stream(log_file)

    def solve_model(self):
        self.model.solve()

    def solve_model_with_callback(self, sub_prob, problem_type):
        # register callback
        callback_instance = self.model.register_callback(LazyConstraintCallbackCplex)
        # create callback attributes
        callback_instance._dam_data = self.dam_data
        callback_instance._sp = sub_prob
        callback_instance._prob = problem_type
        callback_instance._times_called = 0
        callback_instance._cuts_added = 0
        bid_id_2_bbid_var_index = {}
        for bid_id, bbid_var in self.bid_id_2_bbidvar.items():
            index = self.name_2_ind[bbid_var]
            bid_id_2_bbid_var_index[bid_id] = index
        callback_instance._bid_id_2_bbid_var_index = bid_id_2_bbid_var_index
        self.callback_instance = callback_instance
        # turnoff some parameters due to use of callbacks
        # CPLEX automatically makes the below configuration except mip search strategy
        self.model.parameters.threads.set(1)
        self.model.parameters.preprocessing.presolve.set(self.model.parameters.preprocessing.presolve.values.off)
        self.model.parameters.mip.strategy.search.set(self.model.parameters.mip.strategy.search.values.traditional)
        # solve
        self.model.solve()

    def solve_fixed_model(self):
        self.fixed = self.model
        any_interpolated_bids = len(self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid) > 0
        fixed_problem_type = self.fixed.problem_type.fixed_MIQP if any_interpolated_bids else \
            self.fixed.problem_type.fixed_MILP
        self.fixed.set_problem_type(fixed_problem_type)
        self.fixed.solve()

    def solve_relaxed_model(self):
        self.relaxed = cpx.Cplex(self.model)
        self.relaxed.set_problem_type(self.relaxed.problem_type.LP)
        self.relaxed.solve()


class SubProblemCplex(SubProblem):

    def __init__(self, master_problem, working_dir):
        SubProblem.__init__(self, working_dir)
        self.model = cpx.Cplex(master_problem.model)
        self.model.set_problem_type(self.model.problem_type.LP)
        self.write_model()
        # set parameters
        self.model.set_log_stream(None)
        self.model.set_results_stream(None)
        self.objval = None

    def reset_block_bid_bounds(self, block_bids):
        lb = [(ind, 0) for ind in block_bids]
        ub = [(ind, 1) for ind in block_bids]
        self.model.variables.set_lower_bounds(lb)
        self.model.variables.set_upper_bounds(ub)

    def restrict_rejected_block_bids(self, rejected_block_bids):
        # restrict the model
        ub = [(ind, 0) for ind in rejected_block_bids]
        self.model.variables.set_upper_bounds(ub)

    def restrict_accepted_block_bids(self, accepted_block_bids):
        # restrict the model
        lb = [(ind, 1) for ind in accepted_block_bids]
        self.model.variables.set_lower_bounds(lb)

    def write_model(self):
        # write model
        self.model.write(os.path.join(self._working_dir, "sub.lp"))

    def solve_model(self):
        # solve model
        self.model.solve()
        self.objval = self.model.solution.get_objective_value()


class LazyConstraintCallbackCplex(LazyConstraintCallback):

    def __call__(self):
        # update counter
        self._times_called += 1
        bid_id_2_bbid_var_index = self._bid_id_2_bbid_var_index
        sp = self._sp
        prob = self._prob
        # query node obj value
        node_obj = self.get_objective_value()
        # query node solution
        accepted_block_bids = {}
        rejected_block_bids = {}
        for bid_id, ind in bid_id_2_bbid_var_index.items():
            value = self.get_values(ind)
            # TODO: replace the constant with the model parameter value
            if abs(value - 0.0) <= 0.00001:
                rejected_block_bids[bid_id] = ind
            else:
                accepted_block_bids[bid_id] = ind
        # update sub problem
        sp.reset_block_bid_bounds(bid_id_2_bbid_var_index.values())
        if prob is ProblemType.NoPab:
            sp.restrict_rejected_block_bids(rejected_block_bids.values())
        elif prob is ProblemType.NoPrb:
            sp.restrict_accepted_block_bids(accepted_block_bids.values())
        # solve sub problem
        sp.solve_model()
        # assess if the current node solution is valid
        if sp.objval > node_obj + dc.OBJ_COMP_TOL:
            # add lazy constraint to cut this solution
            self._generate_lazy_cuts(accepted_block_bids, rejected_block_bids)

    def _generate_lazy_cuts(self, accepted_block_bids, rejected_block_bids):
        prob_type = self._prob
        self._generate_gcuts(prob_type, accepted_block_bids, rejected_block_bids)

    def _generate_combinatorial_cut_martin(self, accepted_block_bids, rejected_block_bids):
        ind = accepted_block_bids + rejected_block_bids
        coeff = [-1] * len(accepted_block_bids) + [1] * len(rejected_block_bids)
        rhs = 1 - len(accepted_block_bids)
        self.add(constraint=cpx.SparsePair(ind, coeff), sense='G', rhs=rhs)
        self._cuts_added += 1

    def _generate_combinatorial_cut_madani_no_pab(self, accepted_block_bids):
        ind = accepted_block_bids
        coeff = [-1] * len(accepted_block_bids)
        rhs = 1 - len(accepted_block_bids)
        self.add_local(constraint=cpx.SparsePair(ind, coeff), sense='G', rhs=rhs)
        self._cuts_added += 1

    def _generate_combinatorial_cut_madani_no_prb(self, rejected_block_bids):
        ind = rejected_block_bids
        coeff = [1] * len(rejected_block_bids)
        self.add_local(constraint=cpx.SparsePair(ind, coeff), sense='G', rhs=0)
        self._cuts_added += 1

    def _generate_gcuts(self, problem_type, accepted_block_bids, rejected_block_bids):
        bid_id_2_block_bid = self._dam_data.dam_bids.bid_id_2_block_bid
        bid_id_2_bbid_var_index = self._bid_id_2_bbid_var_index
        market_clearing_prices = self._find_market_clearing_prices(
            list(accepted_block_bids.values()), list(rejected_block_bids.values()))
        pabs = du.find_pabs(market_clearing_prices, list(accepted_block_bids.keys()), bid_id_2_block_bid) \
            if problem_type is ProblemType.NoPab else []
        prbs = du.find_prbs(market_clearing_prices, list(rejected_block_bids.keys()), bid_id_2_block_bid) \
            if problem_type is ProblemType.NoPrb else []
        for pab in pabs:
            variables, coefficients, rhs = du.create_gcut_for_pab(
                pab, list(accepted_block_bids.keys()), list(rejected_block_bids.keys()), bid_id_2_block_bid,
                bid_id_2_bbid_var_index)
            self.add(constraint=cpx.SparsePair(variables, coefficients), sense='G', rhs=rhs)
            self._cuts_added += 1
        for prb in prbs:
            variables, coefficients, rhs = du.create_gcut_for_prb(
                prb, list(accepted_block_bids.keys()), list(rejected_block_bids.keys()), bid_id_2_block_bid,
                bid_id_2_bbid_var_index)
            self.add(constraint=cpx.SparsePair(variables, coefficients), sense='G', rhs=rhs)
            self._cuts_added += 1

    def _find_market_clearing_prices(self, accepted_block_bids, rejected_block_bids):
        # solve sub-problem again to obtain dual values
        # restrict accepted and rejected block bids
        sp = self._sp
        dam_data = self._dam_data
        sp.restrict_accepted_block_bids(accepted_block_bids)
        sp.restrict_rejected_block_bids(rejected_block_bids)
        sp.solve_model()
        solution = sp.model.solution
        market_clearing_prices = solution.get_dual_values(
            ['balance_' + str(period) for period in range(1, DamData.NUM_PERIODS + 1, 1)])
        return market_clearing_prices


class BendersDecompositionScip(BendersDecomposition):

    class CallbackData:

        def __init__(self, dam_data=None, bid_id_2_bbidvar=None, sp=None, prob_type=None):
            self._dam_data = dam_data
            self._bid_id_2_bbidvar = bid_id_2_bbidvar
            self._sp = sp
            self._prob_type = prob_type
            self._times_called_lazy = 0
            self._times_added_cut = 0

        def dam_data(self):
            return self._dam_data

        def bid_id_2_bbidvar(self):
            return self._bid_id_2_bbidvar

        def sp(self):
            return self._sp

        def prob_type(self):
            return self._prob_type

        def times_called_lazy(self):
            return self._times_called_lazy

        def times_added_cut(self):
            return self._times_added_cut

        def add_lazy_call(self):
            self._times_called_lazy += 1

        def add_cut(self):
            self._times_added_cut += 1

    def solve(self):
        if self.prob_type is ProblemType.NoPab:
            return self._solve_no_pab()
        elif self.prob_type is ProblemType.NoPrb:
            return self._solve_no_prb()
        elif self.prob_type is ProblemType.Unrestricted:
            return self._solve_unrestricted_problem()
        else:
            raise UnsupportedProblemException(
                "Problem type %s is not supported in method Benders Decomposition of solver Scip" % self.prob_type)

    def _solve_no_pab(self):
        # create master problem
        self.master_problem = MasterProblemScip(self.dam_data, self._working_dir)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)
        # create sub problem
        self.sub_problem = SubProblemScip(self.dam_data, self._working_dir)
        sub_prob = self.sub_problem
        sub_prob.write_model()
        # pass data into callback
        # stores the given info in the 'data' attribute of scip model
        callback_data = BendersDecompositionScip.CallbackData(
            dam_data=self.dam_data, bid_id_2_bbidvar=master_prob.bid_id_2_bbidvar, sp=sub_prob,
            prob_type=ProblemType.NoPab)
        # run benders decomposition
        constraint_handler = LazyConstraintCallbackScip()
        master_prob.solve_model_with_callback(constraint_handler, callback_data)
        return self._get_solver_output()

    def _solve_no_prb(self):
        # create master problem
        self.master_problem = MasterProblemScip(self.dam_data, self._working_dir)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)
        # create sub problem
        self.sub_problem = SubProblemScip(self.dam_data, self._working_dir)
        sub_prob = self.sub_problem
        sub_prob.write_model()
        # pass data into callback
        # stores the given info in the 'data' attribute of scip model
        callback_data = BendersDecompositionScip.CallbackData(
            dam_data=self.dam_data, bid_id_2_bbidvar=master_prob.bid_id_2_bbidvar, sp=sub_prob,
            prob_type=ProblemType.NoPab)
        # run benders decomposition
        constraint_handler = LazyConstraintCallbackScip()
        master_prob.solve_model_with_callback(constraint_handler, callback_data)
        return self._get_solver_output()

    def _solve_unrestricted_problem(self):
        # create master problem
        self.master_problem = MasterProblemScip(self.dam_data, self._working_dir)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)
        master_prob.solve_model()
        return self._get_solver_output()

    def _get_best_solution(self):
        model = self.master_problem.fixed
        # fill dam solution object
        dam_soln = DamSolution()
        dam_soln.total_surplus = model.getObjVal()
        for bid_id, var in self.master_problem.bid_id_2_bbidvar.items():
            value = model.getVal(var)
            if abs(value - 0.0) <= model.getParam('numerics/feastol'):
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        # get market clearing prices by creating a master_problem and solving its relaxation with Gurobi
        # since there are problems with getting dual variable values with scip LP solver
        grb_master_problem = MasterProblemGurobi(self.dam_data, self._working_dir)
        grb_master_problem.create_model()
        grb_master_problem.set_params(self.solver_params)
        for bid_id in dam_soln.rejected_block_bids:
            var = grb_master_problem.bid_id_2_bbidvar.get(bid_id)
            var.ub = 0.0
        for bid_id in dam_soln.accepted_block_bids:
            var = grb_master_problem.bid_id_2_bbidvar.get(bid_id)
            var.lb = 1.0
        # TODO: solving relaxed problem does not work for some reason
        grb_master_problem.solve_model()
        grb_master_problem.solve_fixed_model()
        dam_soln.market_clearing_prices = grb_master_problem.fixed.getAttr(
            'Pi', grb_master_problem.period_2_balance_con.values())
        dam_soln = du.generate_market_result_statistics(self.dam_data.dam_bids, dam_soln)
        return dam_soln

    def _get_solver_output(self):
        # collect optimization stats
        master_problem = self.master_problem
        model = master_problem.model
        elapsed_time = model.getSolvingTime()
        number_of_solutions = len(model.getSols())
        number_of_nodes = model.getNNodes()
        number_of_subproblems_solved = model.data.times_called_lazy() if model.data is not None else 0
        number_of_user_cuts_added = model.data.times_added_cut() if model.data is not None else 0
        benders_stats = BendersDecompositionStats(
            number_of_user_cuts_added=number_of_user_cuts_added,
            number_of_subproblems_solved=number_of_subproblems_solved)
        optimization_stats = OptimizationStats(
            elapsed_time, number_of_nodes, number_of_solutions, benders_decomposition_stats=benders_stats)
        # collect optimization status
        status = model.getStatus()
        best_bound = model.getDualbound()
        mip_relative_gap = model.getGap()
        optimization_status = OptimizationStatus(status, mip_relative_gap, best_bound)
        # best solution query
        if number_of_solutions >= 1:
            master_problem.solve_fixed_model()
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        # clear models
        self.master_problem.model.freeProb()
        if self.sub_problem:
            self.sub_problem.model.freeProb()
        return DamSolverOutput(best_solution, optimization_stats, optimization_status)


class MasterProblemScip(MasterProblem):

    _QUADRATIC_OBJ_EXPR_VAR_NAME = "q_obj"

    def __init__(self, dam_data, working_dir):
        MasterProblem.__init__(self, dam_data, working_dir)
        self.model = scip.Model('master')
        self._quad_obj_var = None

    def _create_hbidvars(self):
        # step hourly bids
        model = self.model
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_simple_bid:
                var_name = 'x_' + str(bid_id) + '_' + str(step_id)
                step_id_2_sbidvar[step_id] = model.addVar(vtype='C', name=var_name, lb=0, ub=1)
            self.bid_id_2_hbidvars[bid_id] = step_id_2_sbidvar
        # piecewise hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_interpolated_bid:
                var_name = 'x_' + str(bid_id) + '_' + str(step_id)
                step_id_2_sbidvar[step_id] = model.addVar(vtype='C', name=var_name, lb=0, ub=1)
            self.bid_id_2_hbidvars[bid_id] = step_id_2_sbidvar

    def _create_bbidvars(self):
        for bid_id in self.dam_data.dam_bids.bid_id_2_block_bid.keys():
            self.bid_id_2_bbidvar[bid_id] = self.model.addVar(vtype='B', name='y_' + str(bid_id))

    def _create_obj_function(self):
        bid_id_2_hbidvars = self.bid_id_2_hbidvars
        bid_id_2_bbidvar = self.bid_id_2_bbidvar
        # set coefficients for simple bids
        lin_vars = []
        lin_coeffs = []
        quad_vars = []
        quad_coeffs = []
        # step hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                lin_vars.append(bid_id_2_hbidvars[bid_id][step_id])
                lin_coeffs.append(simple_bid.p * simple_bid.q)
        # piecewise hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid.items():
            for step_id, interpolated_bid in hourly_bid.step_id_2_interpolated_bid.items():
                lin_vars.append(bid_id_2_hbidvars[bid_id][step_id])
                lin_coeffs.append(interpolated_bid.p_start * interpolated_bid.q)
                quad_vars.append(bid_id_2_hbidvars[bid_id][step_id])
                quad_coeffs.append(0.5 * (interpolated_bid.p_end - interpolated_bid.p_start) * interpolated_bid.q)
        # set coefficients for block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            lin_vars.append(bid_id_2_bbidvar[bid_id])
            lin_coeffs.append(block_bid.price * block_bid.total_quantity)
        obj_expr = [var * coeff for var, coeff in zip(lin_vars, lin_coeffs)]
        # add quadratic objective variable if there are any interpolated bids
        any_interpolated_bids = len(self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid) > 0
        if any_interpolated_bids:
            quad_obj_var = self.model.addVar(vtype='C', name=MasterProblemScip._QUADRATIC_OBJ_EXPR_VAR_NAME, lb=0)
            obj_expr.append(quad_obj_var)
            # add quadratic obj expression as a constraint: this is necessary as SCIP does not support non-linear obj
            self.model.addCons(
                quad_obj_var <= scip.quicksum(var * var * coeff for var, coeff in zip(quad_vars, quad_coeffs)), 
                "quad_obj_con")
        self.model.setObjective(scip.quicksum(obj_expr), 'maximize')

    def _create_balance_constraints(self):
        bid_id_2_hbidvars = self.bid_id_2_hbidvars
        bid_id_2_bbidvar = self.bid_id_2_bbidvar
        period_2_expr = {}
        for period in range(1, DamData.NUM_PERIODS + 1, 1):
            period_2_expr[period] = [[], []]
        # step hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            variables, coeffs = period_2_expr[hourly_bid.period]
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                variables.append(bid_id_2_hbidvars[bid_id][step_id])
                coeffs.append(simple_bid.q)
        # piecewise hourly bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_piecewise_hourly_bid.items():
            variables, coeffs = period_2_expr[hourly_bid.period]
            for step_id, interpolated_bid in hourly_bid.step_id_2_interpolated_bid.items():
                variables.append(bid_id_2_hbidvars[bid_id][step_id])
                coeffs.append(interpolated_bid.q)
        # block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            for t in range(block_bid.period, block_bid.period + block_bid.num_period, 1):
                variables, coeffs = period_2_expr[t]
                variables.append(bid_id_2_bbidvar[bid_id])
                coeffs.append(block_bid.quantity(t))
        for period, expr in period_2_expr.items():
            variables, coeffs = expr
            constraint = self.model.addCons(
                scip.quicksum(var * coeff for var, coeff in zip(variables, coeffs)) == 0.0, 'balance_' + str(period))
            self.period_2_balance_con[period] = constraint

    def _create_cuts_for_identical_bids(self):
        """Sets a complete ordering for the acceptance of identical bid sets
        
        NOTE: Implemented only for block bids"""
        block_bid_lists = self.dam_data.dam_bids.bid_type_2_identical_bid_lists.get(BidType.BLOCK, [])
        model = self.model
        for block_bids in block_bid_lists:
            for index, block_bid in enumerate(block_bids):
                if index == len(block_bids) - 1:
                    continue
                bid_id = block_bid.bid_id
                bid_id_ = block_bids[index + 1].bid_id
                y = self.bid_id_2_bbidvar[bid_id]
                y_ = self.bid_id_2_bbidvar[bid_id_]
                model.addCons(y_ - y <= 0, "identical_bid_ordering_" + bid_id + "_" + bid_id_)

    def create_model(self):
        # create decision variables
        self._create_hbidvars()
        self._create_bbidvars()
        # create objective function
        self._create_obj_function()
        # create constraint set
        self._create_balance_constraints()
        self._create_cuts_for_identical_bids()

    def write_model(self):
        # write model
        self.model.writeProblem(os.path.join(self._working_dir, "master.lp"))

    def set_params(self, solver_params):
        self.model.setRealParam('limits/gap', solver_params.rel_gap)
        self.model.setRealParam('limits/time', solver_params.time_limit)
        # SCIP must be compiled in multi-thread mode to allow multi-thread mip
        self.model.setIntParam('parallel/maxnthreads', solver_params.num_threads)
        # self.model.hideOutput()

    def solve_model(self):
        # solve model
        self.model.optimize()

    def solve_model_with_callback(self, constraint_handler, callback_data):
        """
        Note: 'chckpriority' and 'needscons' parameters must have the specified values.
        'sepapriority' should not be set at all.
        Check https://scip.zib.de/doc/html/CONS.php for info.
        :param constraint_handler:
        :param callback_data
        :return:
        """
        self.model.includeConshdlr(
            constraint_handler, "Lazy", "Constraint handler for Lazy Constraint", sepapriority=0, enfopriority=-1, 
            chckpriority=-1, sepafreq=-1, propfreq=-1, eagerfreq=-1, maxprerounds=0, delaysepa=False, delayprop=False, 
            needscons=False, presoltiming=scip.SCIP_PRESOLTIMING.FAST, proptiming=scip.SCIP_PROPTIMING.BEFORELP)

        self.model.data = callback_data
        self.model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        self.model.setBoolParam("misc/allowstrongdualreds", 0)
        self.model.setBoolParam("misc/allowweakdualreds", 0)
        self.model.optimize()

    def solve_fixed_model(self):
        # get the values of the block bid variables
        bbid_id_2_val = {
            bid_id: self.model.getVal(bbidvar) for bid_id, bbidvar in self.bid_id_2_bbidvar.items()}
        self.fixed = self.model
        self.fixed.freeTransform()
        # convert the problem into fixed MILP/MIQP
        for bid_id, var in self.bid_id_2_bbidvar.items():
            value = bbid_id_2_val[bid_id]
            if abs(value - 0.0) <= self.fixed.getParam('numerics/feastol'):
                self.fixed.chgVarUb(var, 0.0)
            else:
                self.fixed.chgVarLb(var, 1.0)
            self.fixed.chgVarType(var, 'C')
        self.fixed.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        self.fixed.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
        self.fixed.disablePropagation()
        self.fixed.optimize()

    def solve_relaxed_model(self):
        self.relaxed = self.model
        for var in self.bid_id_2_bbidvar.values():
            self.relaxed.chgVarType(var, 'C')
        self.relaxed.optimize()


class SubProblemScip(SubProblem):

    def __init__(self, dam_data, working_dir):
        SubProblem.__init__(self, working_dir)
        # create a copy of the master problem
        copy_master = MasterProblemScip(dam_data, working_dir)
        copy_master.create_model()
        for var in copy_master.bid_id_2_bbidvar.values():
            copy_master.model.chgVarType(var, 'C')
        self.model = copy_master.model
        # assign block bid dictionary to the data member of the subproblem
        self.model.data = copy_master.bid_id_2_bbidvar
        # set the following parameters to get dual information
        self.model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        self.model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
        self.model.disablePropagation()
        self.model.hideOutput()
        self.objval = None

    def reset_block_bid_bounds(self):
        model = self.model
        bid_id_2_bbidvar = model.data
        model.freeTransform()
        for var in bid_id_2_bbidvar.values():
            model.chgVarLb(var, 0.0)
            model.chgVarUb(var, 1.0)

    def restrict_rejected_block_bids(self, rejected_block_bids):
        # restrict the model
        model = self.model
        bid_id_2_bbidvar = model.data
        for bid_id in rejected_block_bids:
            var = bid_id_2_bbidvar[bid_id]
            model.chgVarUb(var, 0.0)

    def restrict_accepted_block_bids(self, accepted_block_bids):
        # restrict the model
        model = self.model
        bid_id_2_bbidvar = model.data
        for bid_id in accepted_block_bids:
            var = bid_id_2_bbidvar[bid_id]
            model.chgVarLb(var, 1.0)

    def write_model(self):
        # write model
        self.model.writeProblem(os.path.join(self._working_dir, "sub.lp"))

    def solve_model(self):
        # solve model
        self.model.optimize()
        self.objval = self.model.getObjVal()


class LazyConstraintCallbackScip(scip.Conshdlr):

    def _add_cut(self, check_only, sol):
        """Do not add cut if this method is called by conscheck(check_only=True)"""
        model = self.model
        callback_data = model.data
        bid_id_2_bbidvar= callback_data.bid_id_2_bbidvar()
        sp = callback_data.sp()
        prob_type = callback_data.prob_type()
        cuts_added = False
        accepted_block_bids = []
        rejected_block_bids = []
        # increase time_called_lazy counter
        callback_data.add_lazy_call()
        node_obj = model.getSolObjVal(sol)
        # query heuristic solution
        for bbid_id, bbidvar in bid_id_2_bbidvar.items():
            value = model.getSolVal(sol, bbidvar)
            if abs(value - 0.0) <= model.getParam('numerics/feastol'):
                rejected_block_bids.append(bbid_id)
            else:
                accepted_block_bids.append(bbid_id)
        # update sub problem
        sp.reset_block_bid_bounds()
        if prob_type is ProblemType.NoPab:
            sp.restrict_rejected_block_bids(rejected_block_bids)
        elif prob_type is ProblemType.NoPrb:
            sp.restrict_accepted_block_bids(accepted_block_bids)
        # solve sub problem
        sp.solve_model()
        # assess if the solution is valid
        if sp.objval > node_obj + dc.OBJ_COMP_TOL:
            cuts_added = True
            if not check_only:
                # add lazy constraint to cut this solution
                self._generate_lazy_cuts(accepted_block_bids, rejected_block_bids, prob_type)
        return cuts_added

    def _generate_lazy_cuts(self, accepted_block_bids, rejected_block_bids, prob):
        self._generate_gcuts(prob, accepted_block_bids, rejected_block_bids)

    def _generate_combinatorial_cut_martin(self, accepted_block_bids, rejected_block_bids):
        model = self.model
        callback_data = model.data
        bid_id_2_bbidvar= callback_data.bid_id_2_bbidvar()
        callback_data.add_lazy_call()
        variables = []
        coeffs = []
        rhs = 1
        for bid_id in accepted_block_bids:
            bid_var = bid_id_2_bbidvar[bid_id]
            variables.append(bid_var)
            coeffs.append(-1)
            rhs -= 1
        for bid_id in rejected_block_bids:
            bid_var = bid_id_2_bbidvar[bid_id]
            variables.append(bid_var)
            coeffs.append(1)
        model.addCons(scip.quicksum(var * coeff for var, coeff in zip(variables, coeffs)) >= rhs)

    def _generate_combinatorial_cut_madani_no_pab(self, accepted_block_bids):
        model = self.model
        callback_data = model.data
        bid_id_2_bbidvar= callback_data.bid_id_2_bbidvar()
        callback_data.add_lazy_call()
        variables = []
        coeffs = []
        rhs = 1
        for bid_id in accepted_block_bids:
            bid_var = bid_id_2_bbidvar[bid_id]
            variables.append(bid_var)
            coeffs.append(-1)
            rhs -= 1
        model.addCons(scip.quicksum(var * coeff for var, coeff in zip(variables, coeffs)) >= rhs, local=True)

    def _generate_combinatorial_cut_madani_no_prb(self, rejected_block_bids):
        model = self.model
        callback_data = model.data
        bid_id_2_bbidvar= callback_data.bid_id_2_bbidvar()
        callback_data.add_lazy_call()
        variables = []
        coeffs = []
        rhs = 1
        for bid_id in rejected_block_bids:
            bid_var = bid_id_2_bbidvar[bid_id]
            variables.append(bid_var)
            coeffs.append(1)
        model.addCons(scip.quicksum(var * coeff for var, coeff in zip(variables, coeffs)) >= rhs, local=True)

    def _generate_gcuts(self, problem_type, accepted_block_bids, rejected_block_bids):
        model = self.model
        callback_data = model.data
        dam_data = callback_data.dam_data()
        bid_id_2_bbidvar= callback_data.bid_id_2_bbidvar()
        callback_data.add_lazy_call()
        bid_id_2_block_bid = dam_data.dam_bids.bid_id_2_block_bid
        market_clearing_prices = self._find_market_clearing_prices(accepted_block_bids, rejected_block_bids)

        pabs = du.find_pabs(market_clearing_prices, accepted_block_bids, bid_id_2_block_bid) \
            if problem_type is ProblemType.NoPab else []
        prbs = du.find_prbs(market_clearing_prices, accepted_block_bids, bid_id_2_block_bid) \
            if problem_type is ProblemType.NoPrb else []
        for pab in pabs:
            variables, coefficients, rhs = du.create_gcut_for_pab(
                pab, accepted_block_bids, rejected_block_bids, bid_id_2_block_bid, bid_id_2_bbidvar)
            model.addCons(scip.quicksum(var * coeff for var, coeff in zip(variables, coefficients)) >= rhs)
            callback_data.add_cut()
        for prb in prbs:
            variables, coefficients, rhs = du.create_gcut_for_prb(
                prb, accepted_block_bids, rejected_block_bids, bid_id_2_block_bid, bid_id_2_bbidvar)
            model.addCons(scip.quicksum(var * coeff for var, coeff in zip(variables, coefficients)) >= rhs)
            callback_data.add_cut()

    def _find_market_clearing_prices(self, accepted_block_bids, rejected_block_bids):
        # solve sub-problem again to obtain dual values
        # restrict accepted and rejected block bids
        model = self.model
        callback_data = model.data
        dam_data = callback_data.dam_data()
        sp = callback_data.sp()
        sp.model.freeTransform()
        sp.restrict_accepted_block_bids(accepted_block_bids)
        sp.restrict_rejected_block_bids(rejected_block_bids)
        sp.solve_model()
        # TODO: replace 'balance_' with a defined constant
        market_clearing_prices = [
            model.getDualsolLinear(con) for con in model.getConss() if con.name.find('balance') != -1]
        return market_clearing_prices[:DamData.NUM_PERIODS]

    def conscheck(self, constraints, solution, check_integrality, check_lp_rows, print_reason, completely):
        if self._add_cut(check_only=True, sol=solution):
            return {"result": scip.SCIP_RESULT.INFEASIBLE}
        else:
            return {"result": scip.SCIP_RESULT.FEASIBLE}

    def consenfolp(self, constraints, n_useful_conss, sol_infeasible):
        if self._add_cut(check_only=False, sol=None):
            # not sure if the result should be 'consadded' or 'separated'
            return {"result": scip.SCIP_RESULT.CONSADDED}
        else:
            return {"result": scip.SCIP_RESULT.FEASIBLE}

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
        pass
