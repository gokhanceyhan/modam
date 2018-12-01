from abc import abstractmethod
import gurobipy as grb
import cplex as cpx
from pyscipopt import Model
import dam_utils as du
import dam_solver as ds


class PrimalDualModel:
    def __init__(self, prob_type, dam_data, prob_name):
        self.prob_type = prob_type
        self.dam_data = dam_data
        self.prob_name = prob_name
        self.model = grb.Model(prob_name)
        self.bid_id_2_step_id_2_sbidvar = {}
        self.bid_id_2_bbidvar = {}
        self.period_2_balance_con = {}
        self.period_2_pi = {}

    def create_model(self):
        self._create_variables()
        self._create_obj_function()
        self._create_constraints()
        self._write_model()
        return self.prob_name + '.mps'

    def _write_model(self):
        self.model.write(self.prob_name + '.mps')

    def _create_variables(self):
        self._create_hbidvars()
        self._create_bbidvars()
        self._create_price_variables()

    def _create_constraints(self):
        self._create_balance_constraints()
        self._create_dual_feasibility_constraints()
        self._restrict_loss_variables()
        self._create_strong_duality_constraint()

    def _create_hbidvars(self):
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_simple_bid.keys():
                pvar = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                         name='x_' + str(bid_id) + '_' + str(step_id), lb=0, ub=1)
                dvar = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                         name='s_' + str(bid_id) + '_' + str(step_id), lb=0)
                step_id_2_sbidvar[step_id] = (pvar, dvar)
            self.bid_id_2_step_id_2_sbidvar[bid_id] = step_id_2_sbidvar

    def _create_bbidvars(self):
        for bid_id in self.dam_data.dam_bids.bid_id_2_block_bid.keys():
            pvar = self.model.addVar(vtype=grb.GRB.BINARY, name='y_' + str(bid_id))
            dvar = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='s_' + str(bid_id), lb=0)
            lvar = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='l_' + str(bid_id), lb=0)
            mvar = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='m_' + str(bid_id), lb=0)
            if self.prob_type is ds.ProblemType.NoPab:
                lvar.ub = 0
            if self.prob_type is ds.ProblemType.NoPrb:
                mvar.ub = 0
            self.bid_id_2_bbidvar[bid_id] = (pvar, dvar, lvar, mvar)

    def _create_price_variables(self):
        for period in range(self.dam_data.number_of_periods):
            var = self.model.addVar(vtype=grb.GRB.CONTINUOUS, name='pi_' + str(period+1), lb=self.dam_data.min_price,
                                    ub=self.dam_data.max_price)
            self.period_2_pi[period+1] = var

    def _create_obj_function(self):
        lin_expr = grb.LinExpr(0.0)
        # set coefficients for simple bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                lin_expr.add(self.bid_id_2_step_id_2_sbidvar[bid_id][step_id][0], step[0] * step[1])
        # set coefficients for block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            lin_expr.add(self.bid_id_2_bbidvar[bid_id][0], block_bid.num_period * block_bid.price * block_bid.quantity)
        self.model.setObjective(lin_expr, grb.GRB.MAXIMIZE)

    def _create_balance_constraints(self):
        period_2_expr = {}
        for period in range(1, self.dam_data.number_of_periods + 1, 1):
            expr = grb.LinExpr(0.0)
            period_2_expr[period] = expr
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                expr = period_2_expr[hourly_bid.period]
                expr.add(self.bid_id_2_step_id_2_sbidvar[bid_id][step_id][0], step[1])
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
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                expr = grb.LinExpr(0.0)
                svar = bid_id_2_step_id_2_sbidvar[bid_id][step_id][1]
                pi = period_2_pi[hourly_bid.period]
                p = step[0]
                q = step[1]
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

    def _restrict_loss_variables(self):
        bid_id_2_bbidvar = self.bid_id_2_bbidvar
        model = self.model
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            mexpr = grb.LinExpr(0.0)
            lexpr = grb.LinExpr(0.0)
            y, s, l, m = bid_id_2_bbidvar[bid_id]
            bigm = du.calculate_bigm_for_block_bid_loss(block_bid)
            mexpr.addTerms([1, bigm], [m, y])
            lexpr.addTerms([1, -bigm], [l, y])
            model.addConstr(mexpr, grb.GRB.LESS_EQUAL, bigm, 'missed_surplus_' + str(bid_id))
            model.addConstr(lexpr, grb.GRB.LESS_EQUAL, 0.0, 'loss_surplus_' + str(bid_id))

    def _create_strong_duality_constraint(self):
        bid_id_2_step_id_2_sbidvar = self.bid_id_2_step_id_2_sbidvar
        bid_id_2_bbidvar = self.bid_id_2_bbidvar
        model = self.model

        lin_expr = grb.LinExpr(0.0)
        # set coefficients for simple bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                pvar, dvar = bid_id_2_step_id_2_sbidvar[bid_id][step_id]
                lin_expr.add(pvar, step[0] * step[1])
                lin_expr.add(dvar, -1)
        # set coefficients for block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            y, s, l, m = bid_id_2_bbidvar[bid_id]
            lin_expr.add(y, block_bid.num_period * block_bid.price * block_bid.quantity)
            lin_expr.addTerms([-1, 1], [s, l])
        model.addConstr(lin_expr, grb.GRB.GREATER_EQUAL, 0.0, 'sd')


class PrimalDualSolver(object):
    def __init__(self, prob_name, solver_params):
        self.prob_name = prob_name
        self.solver_params = solver_params

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
    def __init__(self, prob_name, solver_params):
        PrimalDualSolver.__init__(self, prob_name, solver_params)
        self.model = grb.read(prob_name)

    def solve(self):
        self._set_params()
        self.model.optimize()
        return self._get_solver_output()

    def _set_params(self):
        self.model.Params.LogToConsole = 0
        self.model.Params.MIPGap = self.solver_params.rel_gap
        self.model.Params.TimeLimit = self.solver_params.time_limit

    def _get_best_solution(self):
        # fill solution
        dam_soln = ds.DamSolution()
        dam_soln.total_surplus = self.model.ObjVal
        varname_2_bbidvar = {x.VarName: x for x in self.model.getVars() if x.VarName.find('y') != -1}
        y = self.model.getAttr('X', varname_2_bbidvar)
        for name, value in y.items():
            bid_id = name[2:]
            if abs(value - 0.0) <= self.model.Params.IntFeasTol:
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        dam_soln.market_clearing_prices = [x.X for x in self.model.getVars() if x.VarName.find('pi') != -1]
        return dam_soln

    def _get_solver_output(self):
        # collect optimization stats
        elapsed_time = self.model.Runtime
        number_of_solutions = self.model.SolCount
        number_of_nodes = self.model.NodeCount
        optimization_stats = ds.OptimizationStats(elapsed_time, number_of_nodes, number_of_solutions)
        # collect optimization status
        status = self.model.Status
        best_bound = self.model.ObjBound
        mip_relative_gap = self.model.MIPGap
        optimization_status = ds.OptimizationStatus(status, mip_relative_gap, best_bound)
        # best solution query
        if number_of_solutions >= 1:
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        return ds.DamSolverOutput(best_solution, optimization_stats, optimization_status)


class PrimalDualCplexSolver(PrimalDualSolver):
    def __init__(self, prob_name, solver_params):
        PrimalDualSolver.__init__(self, prob_name, solver_params)
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
        self.model.set_log_stream('cplex.log')
        self.model.set_results_stream('cplex.log')
        self.model.set_warning_stream('cplex.log')

    def _get_best_solution(self):
        solution = self.model.solution
        # fill dam solution object
        dam_soln = ds.DamSolution()
        dam_soln.total_surplus = solution.get_objective_value()
        bbid_varnames = [name for name in self.model.variables.get_names() if name.find('y') != -1]
        varname_2_y = {name: solution.get_values(name) for name in bbid_varnames}
        for name, y in varname_2_y.items():
            bid_id = name[2:]
            if abs(y - 0.0) <= self.model.parameters.mip.tolerances.integrality.get():
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        pi_varnames = [name for name in self.model.variables.get_names() if name.find('pi') != -1]
        pi = solution.get_values(pi_varnames)
        dam_soln.market_clearing_prices = pi
        return dam_soln

    def _get_solver_output(self, elapsed_time):
        solution = self.model.solution
        # collect optimization stats
        elapsed_time = elapsed_time
        number_of_solutions = solution.pool.get_num()
        number_of_nodes = solution.progress.get_num_nodes_processed()
        optimization_stats = ds.OptimizationStats(elapsed_time, number_of_nodes, number_of_solutions)
        # collect optimization status
        status = solution.get_status()
        best_bound = solution.MIP.get_best_objective()
        mip_relative_gap = solution.MIP.get_mip_relative_gap()
        optimization_status = ds.OptimizationStatus(status, mip_relative_gap, best_bound)
        # best solution query
        if number_of_solutions >= 1:
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        return ds.DamSolverOutput(best_solution, optimization_stats, optimization_status)


class PrimalDualScipSolver(PrimalDualSolver):
    def __init__(self, prob_name, solver_params):
        PrimalDualSolver.__init__(self, prob_name, solver_params)
        self.model = Model()
        self.model.readProblem(prob_name)

    def solve(self):
        self._set_params()
        self.model.optimize()
        return self._get_solver_output()

    def _set_params(self):
        self.model.setRealParam('limits/gap', self.solver_params.rel_gap)
        self.model.setRealParam('limits/time', self.solver_params.time_limit)
        # self.model.hideOutput()

    def _get_best_solution(self):
        model = self.model
        # fill dam solution object
        dam_soln = ds.DamSolution()
        dam_soln.total_surplus = model.getObjVal()
        varname_2_bbidvar = {x.name: x for x in model.getVars() if x.name.find('y') != -1}
        varname_2_y = {name: model.getVal(var) for name, var in varname_2_bbidvar.items()}
        for name, value in varname_2_y.items():
            bid_id = name[2:]
            if abs(value - 0.0) <= model.getParam('numerics/feastol'):
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        dam_soln.market_clearing_prices = [model.getVal(x) for x in model.getVars() if x.name.find('pi') != -1]
        return dam_soln

    def _get_solver_output(self):
        # collect optimization stats
        model = self.model
        elapsed_time = model.getSolvingTime()
        number_of_solutions = len(model.getSols())
        number_of_nodes = model.getNNodes()
        optimization_stats = ds.OptimizationStats(elapsed_time, number_of_nodes, number_of_solutions)
        # collect optimization status
        status = model.getStatus()
        best_bound = model.getDualbound()
        mip_relative_gap = model.getGap()
        optimization_status = ds.OptimizationStatus(status, mip_relative_gap, best_bound)
        # best solution query
        if number_of_solutions >= 1:
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        return ds.DamSolverOutput(best_solution, optimization_stats, optimization_status)
