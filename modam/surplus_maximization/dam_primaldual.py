from abc import abstractmethod
import os

import gurobipy as grb
import cplex as cpx
from pyscipopt import Model

from modam.surplus_maximization.dam_common import DamSolution, DamSolverOutput, OptimizationStats, OptimizationStatus, \
    ProblemType
from modam.surplus_maximization.dam_input import BidType, DamData
from modam.surplus_maximization.dam_utils import calculate_bigm_for_block_bid_loss, \
    calculate_bigm_for_block_bid_missed_surplus


class PrimalDualModel:

    def __init__(self, prob_type, dam_data, prob_name, working_dir):
        self.prob_type = prob_type
        self.dam_data = dam_data
        self.prob_name = prob_name
        self.model = grb.Model(prob_name)
        self.bid_id_2_step_id_2_sbidvar = {}
        self.bid_id_2_bbidvar = {}
        self.period_2_balance_con = {}
        self.period_2_pi = {}
        self._working_dir = working_dir
        self.loss_expr = grb.LinExpr(0.0)
        self.missed_surplus_expr = grb.LinExpr(0.0)
        self.surplus_expr = grb.LinExpr(0.0)

    def create_model(self):
        self._create_variables()
        self._create_obj_function()
        self._create_constraints()
        self._write_model()
        return os.path.join(self._working_dir, self.prob_name + '.mps')

    def _write_model(self):
        self.model.write(os.path.join(self._working_dir, self.prob_name + '.mps'))

    def _create_variables(self):
        self._create_hbidvars()
        self._create_bbidvars()
        self._create_price_variables()

    def _create_constraints(self):
        self._create_balance_constraints()
        self._create_dual_feasibility_constraints()
        self._restrict_loss_variables()
        self._create_strong_duality_constraint()
        self._create_cuts_for_identical_bids()

    def _create_hbidvars(self):
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_simple_bid.keys():
                pvar = self.model.addVar(
                    vtype=grb.GRB.CONTINUOUS, name='x_' + str(bid_id) + '_' + str(step_id), lb=0, ub=1)
                dvar = self.model.addVar(
                    vtype=grb.GRB.CONTINUOUS, name='s_' + str(bid_id) + '_' + str(step_id), lb=0)
                step_id_2_sbidvar[step_id] = (pvar, dvar)
            self.bid_id_2_step_id_2_sbidvar[bid_id] = step_id_2_sbidvar

    def _create_bbidvars(self):
        for bid_id in self.dam_data.dam_bids.bid_id_2_block_bid.keys():
            pvar = self.model.addVar(vtype=grb.GRB.BINARY, name='y_' + str(bid_id))
            dvar = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='s_' + str(bid_id), lb=0)
            lvar = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='l_' + str(bid_id), lb=0)
            mvar = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='m_' + str(bid_id), lb=0)
            if self.prob_type is ProblemType.NoPab:
                lvar.ub = 0
            if self.prob_type is ProblemType.NoPrb:
                mvar.ub = 0
            self.bid_id_2_bbidvar[bid_id] = (pvar, dvar, lvar, mvar)

    def _create_price_variables(self):
        for period in range(DamData.NUM_PERIODS):
            var = self.model.addVar(
                vtype=grb.GRB.CONTINUOUS, name='pi_' + str(period+1), lb=DamData.MIN_PRICE, ub=DamData.MAX_PRICE)
            self.period_2_pi[period+1] = var

    def _create_obj_function(self):
        lin_expr = grb.LinExpr(0.0)
        # set coefficients for simple bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                lin_expr.add(self.bid_id_2_step_id_2_sbidvar[bid_id][step_id][0], simple_bid.p * simple_bid.q)
        # set coefficients for block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            lin_expr.add(self.bid_id_2_bbidvar[bid_id][0], block_bid.num_period * block_bid.price * block_bid.quantity)
        self.model.setObjective(lin_expr, grb.GRB.MAXIMIZE)
        self.surplus_expr.add(lin_expr)

    def _create_balance_constraints(self):
        period_2_expr = {}
        for period in range(1, DamData.NUM_PERIODS + 1, 1):
            expr = grb.LinExpr(0.0)
            period_2_expr[period] = expr
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                expr = period_2_expr[hourly_bid.period]
                expr.add(self.bid_id_2_step_id_2_sbidvar[bid_id][step_id][0], simple_bid.q)
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            for t in range(block_bid.period, block_bid.period + block_bid.num_period, 1):
                expr = period_2_expr[t]
                expr.add(self.bid_id_2_bbidvar[bid_id][0], block_bid.quantity)
        for period, expr in period_2_expr.items():
            constraint = self.model.addConstr(expr, grb.GRB.EQUAL, 0.0, 'balance_' + str(period))
            self.period_2_balance_con[period] = constraint

    def _create_dual_feasibility_constraints(self):
        bid_id_2_step_id_2_sbidvar = self.bid_id_2_step_id_2_sbidvar
        bid_id_2_bbidvar = self.bid_id_2_bbidvar
        model = self.model
        period_2_pi = self.period_2_pi
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                expr = grb.LinExpr(0.0)
                svar = bid_id_2_step_id_2_sbidvar[bid_id][step_id][1]
                pi = period_2_pi[hourly_bid.period]
                p = simple_bid.p
                q = simple_bid.q
                expr.addTerms([1, q], [svar, pi])
                model.addConstr(expr, grb.GRB.GREATER_EQUAL, p*q, 'dual_' + str(bid_id) + '_' + str(step_id))

        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            expr = grb.LinExpr(0.0)
            y, s, l, m = bid_id_2_bbidvar[bid_id]
            p = block_bid.num_period * [block_bid.price]
            q = block_bid.num_period * [block_bid.quantity]
            pi = list(period_2_pi.values())
            pi = pi[block_bid.period-1:block_bid.period+block_bid.num_period-1]
            rhs = sum([i*j for i, j in zip(p, q)])
            expr.addTerms([1, -1, 1], [s, l, m])
            expr.addTerms(q, pi)
            model.addConstr(expr, grb.GRB.GREATER_EQUAL, rhs, 'dual_' + str(bid_id))

    def _create_strong_duality_constraint(self):
        bid_id_2_step_id_2_sbidvar = self.bid_id_2_step_id_2_sbidvar
        bid_id_2_bbidvar = self.bid_id_2_bbidvar
        model = self.model

        lin_expr = grb.LinExpr(0.0)
        # set coefficients for simple bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_step_hourly_bid.items():
            for step_id, simple_bid in hourly_bid.step_id_2_simple_bid.items():
                pvar, dvar = bid_id_2_step_id_2_sbidvar[bid_id][step_id]
                lin_expr.add(pvar, simple_bid.p * simple_bid.q)
                lin_expr.add(dvar, -1)
        # set coefficients for block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            y, s, l, m = bid_id_2_bbidvar[bid_id]
            lin_expr.add(y, block_bid.num_period * block_bid.price * block_bid.quantity)
            lin_expr.addTerms([-1, 1], [s, l])
        model.addConstr(lin_expr, grb.GRB.GREATER_EQUAL, 0.0, 'sd')

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
                y = self.bid_id_2_bbidvar[bid_id][0]
                y_ = self.bid_id_2_bbidvar[bid_id_][0]
                model.addConstr(y_ - y, grb.GRB.LESS_EQUAL, 0, "identical_bid_ordering_" + bid_id + "_" + bid_id_)

    def _restrict_loss_variables(self):
        bid_id_2_bbidvar = self.bid_id_2_bbidvar
        model = self.model
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            mexpr = grb.LinExpr(0.0)
            lexpr = grb.LinExpr(0.0)
            y, s, l, m = bid_id_2_bbidvar[bid_id]
            l_bigm = calculate_bigm_for_block_bid_loss(
                block_bid, max_price=DamData.MAX_PRICE, min_price=DamData.MIN_PRICE)
            m_bigm = calculate_bigm_for_block_bid_missed_surplus(
                block_bid, max_price=DamData.MAX_PRICE, min_price=DamData.MIN_PRICE)
            mexpr.addTerms([1, m_bigm], [m, y])
            lexpr.addTerms([1, -l_bigm], [l, y])
            model.addConstr(mexpr, grb.GRB.LESS_EQUAL, m_bigm, 'missed_surplus_' + str(bid_id))
            model.addConstr(lexpr, grb.GRB.LESS_EQUAL, 0.0, 'loss_surplus_' + str(bid_id))
            self.loss_expr.add(l)
            self.missed_surplus_expr.add(m)


class PrimalDualSolver(object):

    def __init__(self, prob_name, solver_params, working_dir):
        self.prob_name = prob_name
        self.solver_params = solver_params
        self._working_dir = working_dir

    @abstractmethod
    def _set_params(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def _get_best_solution(self):
        pass

    @abstractmethod
    def _get_solver_output(self, *args):
        pass


class PrimalDualGurobiSolver(PrimalDualSolver):

    def __init__(self, prob_name, solver_params, working_dir):
        PrimalDualSolver.__init__(self, prob_name, solver_params, working_dir)
        self.model = grb.read(prob_name)

    def solve(self):
        self._set_params()
        self.model.optimize()
        return self._get_solver_output()

    def _set_params(self):
        self.model.Params.LogToConsole = 0
        self.model.Params.MIPGap = self.solver_params.rel_gap
        self.model.Params.TimeLimit = self.solver_params.time_limit
        self.model.Params.Threads = self.solver_params.num_threads

    def _get_best_solution(self):
        # fill solution
        dam_soln = DamSolution()
        dam_soln.total_surplus = -1 * self.model.ObjVal
        varname_2_bbidvar = {x.VarName: x for x in self.model.getVars() if x.VarName.find('y_') == 0}
        y = self.model.getAttr('X', varname_2_bbidvar)
        for name, value in y.items():
            bid_id = name[2:]
            if abs(value - 0.0) <= self.model.Params.IntFeasTol:
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        dam_soln.market_clearing_prices = [x.X for x in self.model.getVars() if x.VarName.find('pi_') == 0]
        return dam_soln

    def _get_solver_output(self):
        # collect optimization stats
        elapsed_time = self.model.Runtime
        number_of_solutions = self.model.SolCount
        number_of_nodes = self.model.NodeCount
        optimization_stats = OptimizationStats(elapsed_time, number_of_nodes, number_of_solutions)
        # collect optimization status
        status = self.model.Status
        best_bound = self.model.ObjBound
        mip_relative_gap = self.model.MIPGap
        optimization_status = OptimizationStatus(status, mip_relative_gap, best_bound)
        # best solution query
        if number_of_solutions >= 1:
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        return DamSolverOutput(best_solution, optimization_stats, optimization_status)


class PrimalDualCplexSolver(PrimalDualSolver):

    def __init__(self, prob_name, solver_params, working_dir):
        PrimalDualSolver.__init__(self, prob_name, solver_params, working_dir)
        self.model = cpx.Cplex(prob_name)
        self.model.read(prob_name)

    def solve(self):
        self._set_params()
        start_time = self.model.get_time()
        self.model.solve()
        end_time = self.model.get_time()
        elapsed_time = end_time - start_time
        return self._get_solver_output(elapsed_time)

    def _set_params(self):
        self.model.parameters.mip.tolerances.mipgap.set(self.solver_params.rel_gap)
        self.model.parameters.timelimit.set(self.solver_params.time_limit)
        self.model.parameters.threads.set(self.solver_params.num_threads)
        log_file = os.path.join(self._working_dir, 'cplex.log')
        self.model.set_log_stream(log_file)
        self.model.set_results_stream(log_file)
        self.model.set_warning_stream(log_file)

    def _get_best_solution(self):
        solution = self.model.solution
        # fill dam solution object
        dam_soln = DamSolution()
        dam_soln.total_surplus = -1 * solution.get_objective_value()
        bbid_varnames = [name for name in self.model.variables.get_names() if name.find('y_') == 0]
        varname_2_y = {name: solution.get_values(name) for name in bbid_varnames}
        for name, y in varname_2_y.items():
            bid_id = name[2:]
            if abs(y - 0.0) <= self.model.parameters.mip.tolerances.integrality.get():
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        pi_varnames = [name for name in self.model.variables.get_names() if name.find('pi_') == 0]
        pi = solution.get_values(pi_varnames)
        dam_soln.market_clearing_prices = pi
        return dam_soln

    def _get_solver_output(self, elapsed_time):
        solution = self.model.solution
        # collect optimization stats
        elapsed_time = elapsed_time
        number_of_solutions = solution.pool.get_num()
        number_of_nodes = solution.progress.get_num_nodes_processed()
        optimization_stats = OptimizationStats(elapsed_time, number_of_nodes, number_of_solutions)
        # collect optimization status
        status = solution.get_status()
        best_bound = solution.MIP.get_best_objective()
        mip_relative_gap = solution.MIP.get_mip_relative_gap() if number_of_solutions >= 1 else -1
        optimization_status = OptimizationStatus(status, mip_relative_gap, best_bound)
        # best solution query
        if number_of_solutions >= 1:
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        return DamSolverOutput(best_solution, optimization_stats, optimization_status)


class PrimalDualScipSolver(PrimalDualSolver):

    def __init__(self, prob_name, solver_params, working_dir):
        PrimalDualSolver.__init__(self, prob_name, solver_params, working_dir)
        self.model = Model()
        self.model.readProblem(prob_name)

    def solve(self):
        self._set_params()
        self.model.optimize()
        return self._get_solver_output()

    def _set_params(self):
        self.model.setRealParam('limits/gap', self.solver_params.rel_gap)
        self.model.setRealParam('limits/time', self.solver_params.time_limit)
        # SCIP must be compiled in multi-thread mode to allow multi-thread mip
        self.model.setIntParam('parallel/maxnthreads', self.solver_params.num_threads)
        self.model.hideOutput()

    def _get_best_solution(self):
        model = self.model
        # fill dam solution object
        dam_soln = DamSolution()
        dam_soln.total_surplus = -1 * model.getObjVal()
        varname_2_bbidvar = {x.name: x for x in model.getVars() if x.name.find('y_') == 0}
        varname_2_y = {name: model.getVal(var) for name, var in varname_2_bbidvar.items()}
        for name, value in varname_2_y.items():
            bid_id = name[2:]
            if abs(value - 0.0) <= model.getParam('numerics/feastol'):
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        dam_soln.market_clearing_prices = [model.getVal(x) for x in model.getVars() if x.name.find('pi_') == 0]
        return dam_soln

    def _get_solver_output(self):
        # collect optimization stats
        model = self.model
        elapsed_time = model.getSolvingTime()
        number_of_solutions = len(model.getSols())
        number_of_nodes = model.getNNodes()
        optimization_stats = OptimizationStats(elapsed_time, number_of_nodes, number_of_solutions)
        # collect optimization status
        status = model.getStatus()
        best_bound = model.getDualbound()
        mip_relative_gap = model.getGap()
        optimization_status = OptimizationStatus(status, mip_relative_gap, best_bound)
        # best solution query
        if number_of_solutions >= 1:
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        return DamSolverOutput(best_solution, optimization_stats, optimization_status)
