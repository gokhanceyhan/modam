from abc import abstractmethod

import gurobipy as grb
import cplex as cpx
from cplex.callbacks import LazyConstraintCallback

import dam_constants as dc
import dam_solver as ds


class BendersDecomposition(object):
    def __init__(self, prob_type, dam_data, solver_params):
        self.prob_type = prob_type
        self.dam_data = dam_data
        self.solver_params = solver_params
        self.master_problem = None
        self.sub_problem = None
        self.callback = None

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def _solve_no_pab(self):
        pass

    @abstractmethod
    def _solve_no_prb(self):
        pass


class MasterProblem(object):
    def __init__(self, dam_data):
        self.dam_data = dam_data
        self.model = None
        self.fixed = None
        self.relaxed = None
        self.bid_id_2_hbidvars = {}
        self.bid_id_2_bbidvar = {}
        self.period_2_balance_con = {}

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
    def __init__(self, *args):
        pass

    @abstractmethod
    def reset_block_bid_bounds(self, bid_id_2_bbidvar):
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
    def __init__(self, prob_type, dam_data, solver_params):
        BendersDecomposition.__init__(self, prob_type, dam_data, solver_params)

    def solve(self):
        if self.prob_type is ds.ProblemType.NoPab:
            self._solve_no_pab()

    def _solve_no_pab(self):
        # create master problem
        self.master_problem = MasterProblemGurobi(self.dam_data)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)

        # create sub problem
        self.sub_problem = SubProblemGurobi(master_prob)
        sub_prob = self.sub_problem

        # pass data into callback
        master_prob.model._bid_id_2_bbidvar = master_prob.bid_id_2_bbidvar
        master_prob.model._sp = sub_prob
        master_prob.model._prob = ds.ProblemType.NoPab

        # run benders decomposition
        callback = CallbackGurobi()
        master_prob.solve_model_with_callback(callback.dam_callback)
        master_prob.solve_fixed_model()

    def _solve_no_prb(self):
        pass


class MasterProblemGurobi(MasterProblem):
    def __init__(self, dam_data):
        MasterProblem.__init__(self, dam_data)
        self.model = grb.Model('master')

    def _create_hbidvars(self):
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            step_id_2_sbidvars = {}
            for step_id in hourly_bid.step_id_2_simple_bid.keys():
                step_id_2_sbidvars[step_id] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                                                name='x_' + str(bid_id) + '_' + str(step_id), lb=0,
                                                                ub=1)
            self.bid_id_2_hbidvars[bid_id] = step_id_2_sbidvars

    def _create_bbidvars(self):
        for bid_id in self.dam_data.dam_bids.bid_id_2_block_bid.keys():
            self.bid_id_2_bbidvar[bid_id] = self.model.addVar(vtype=grb.GRB.BINARY, name='y_' + str(bid_id))

    def _create_obj_function(self):
        lin_expr = grb.LinExpr(0.0)
        # set coefficients for simple bids
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                lin_expr.add(self.bid_id_2_hbidvars[bid_id][step_id], step[0] * step[1])
        # set coefficients for block bids
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            lin_expr.add(self.bid_id_2_bbidvar[bid_id], block_bid.num_period * block_bid.price * block_bid.quantity)
        self.model.setObjective(lin_expr, grb.GRB.MAXIMIZE)

    def _create_balance_constraints(self):
        period_2_expr = {}
        for period in range(1, self.dam_data.number_of_periods + 1, 1):
            expr = grb.LinExpr(0.0)
            period_2_expr[period] = expr
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                expr = period_2_expr[hourly_bid.period]
                expr.add(self.bid_id_2_hbidvars[bid_id][step_id], step[1])
        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            for t in range(block_bid.period, block_bid.period + block_bid.num_period, 1):
                expr = period_2_expr[t]
                expr.add(self.bid_id_2_bbidvar[bid_id], block_bid.quantity)
        for period, expr in period_2_expr.items():
            constraint = self.model.addConstr(expr, grb.GRB.EQUAL, 0.0, 'balance_' + str(period))
            self.period_2_balance_con[period] = constraint

    def create_model(self):
        # create decision variables
        self._create_hbidvars()
        self._create_bbidvars()
        # create objective function
        self._create_obj_function()
        # create constraint set
        self._create_balance_constraints()

    def write_model(self):
        # write model
        self.model.write('master.lp')

    def set_params(self, solver_params):
        self.model.Params.LogToConsole = 0
        self.model.Params.MIPGap = solver_params.rel_gap
        self.model.Params.TimeLimit = solver_params.time_limit

    def solve_model(self):
        # solve model
        self.model.optimize()

    def solve_model_with_callback(self, callback):
        self.model.Params.Heuristics = 0
        self.model.Params.LazyConstraints = 1
        self.model.optimize(callback)

    def solve_fixed_model(self):
        # solve restricted lp model
        self.fixed = self.model.fixed()
        self.fixed.optimize()

    def solve_relaxed_model(self):
        self.relaxed = self.model.relax()
        self.relaxed.optimize()


class SubProblemGurobi(SubProblem):
    def __init__(self, master_problem):
        SubProblem.__init__(self)
        self.model = master_problem.model.copy().relax()
        self.objval = None

    def reset_block_bid_bounds(self, bid_id_2_bbidvar):
        for bid_id in bid_id_2_bbidvar.keys():
            var = self.model.getVarByName('y_' + str(bid_id))
            var.lb = 0.0
            var.ub = 1.0

    def restrict_rejected_block_bids(self, rejected_block_bids):
        # restrict the model
        for bid_id in rejected_block_bids:
            var = self.model.getVarByName('y_' + str(bid_id))
            var.ub = 0.0

    def restrict_accepted_block_bids(self, accepted_block_bids):
        # restrict the model
        for bid_id in accepted_block_bids:
            var = self.model.getVarByName('y_' + str(bid_id))
            var.lb = 1.0

    def write_model(self):
        # write model
        self.model.write('sub.lp')

    def solve_model(self):
        # set parameters
        self.model.Params.OutputFlag = 0
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
            if model._prob is ds.ProblemType.NoPab:
                model._sp.restrict_rejected_block_bids(rejected_block_bids)
            elif model._prob is ds.ProblemType.NoPrb:
                model._sp.restrict_accepted_block_bids(accepted_block_bids)
            # solve sub problem
            model._sp.solve_model()
            # assess if the current node solution is valid
            if model._sp.objval > node_obj + dc.OBJ_COMP_TOL:
                # add lazy constraint to cut this solution
                CallbackGurobi._generate_lazy_cuts(model, accepted_block_bids, rejected_block_bids,
                                                   model._bid_id_2_bbidvar)

    @staticmethod
    def _generate_lazy_cuts(model, accepted_block_bids, rejected_block_bids, bid_id_2_bbidvar):
        CallbackGurobi._generate_combinatorial_cut_martin(model, accepted_block_bids, rejected_block_bids,
                                                          bid_id_2_bbidvar)

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


class BendersDecompositionCplex(BendersDecomposition):
    def __init__(self, prob_type, dam_data, solver_params):
        BendersDecomposition.__init__(self, prob_type, dam_data, solver_params)

    def solve(self):
        if self.prob_type is ds.ProblemType.NoPab:
            self._solve_no_pab()

    def _solve_no_pab(self):
        # create master problem
        self.master_problem = MasterProblemCplex(self.dam_data)
        master_prob = self.master_problem
        master_prob.create_model()
        master_prob.write_model()
        master_prob.set_params(self.solver_params)

        # create sub problem
        self.sub_problem = SubProblemCplex(master_prob)
        sub_prob = self.sub_problem

        # run benders decomposition
        master_prob.solve_model_with_callback(sub_prob)
        master_prob.solve_fixed_model()

    def _solve_no_prb(self):
        pass


class MasterProblemCplex(MasterProblem):
    def __init__(self, dam_data):
        MasterProblem.__init__(self, dam_data)
        self.model = cpx.Cplex()
        self.name_2_ind = None

    def _create_hbidvars(self):
        model = self.model
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            step_id_2_sbidvars = {}
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                var_name = 'x_' + str(bid_id) + '_' + str(step_id)
                obj_coeff = step[0]*step[1]
                model.variables.add(obj=[obj_coeff], lb=[0.0], ub=[1.0], types=['C'], names=[var_name])
                step_id_2_sbidvars[step_id] = var_name
            self.bid_id_2_hbidvars[bid_id] = step_id_2_sbidvars

    def _create_bbidvars(self):
        for bid_id, bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            var_name = 'y_' + str(bid_id)
            obj_coeff = bid.price * bid.quantity * bid.num_period
            self.model.variables.add(obj=[obj_coeff], lb=[0.0], ub=[1.0], types=['B'], names=[var_name])
            self.bid_id_2_bbidvar[bid_id] = var_name

    def _create_obj_function(self):
        self.model.objective.set_sense(self.model.objective.sense.maximize)

    def _create_balance_constraints(self):
        inds = [[] for i in range(self.dam_data.number_of_periods)]
        vals = [[] for i in range(self.dam_data.number_of_periods)]

        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            period = hourly_bid.period
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                var_name = self.bid_id_2_hbidvars[bid_id][step_id]
                inds[period-1].append(self.model.variables.get_indices(var_name))
                vals[period-1].append(step[1])

        for bid_id, block_bid in self.dam_data.dam_bids.bid_id_2_block_bid.items():
            var_name = self.bid_id_2_bbidvar[bid_id]
            ind = self.model.variables.get_indices(var_name)
            for t in range(block_bid.period, block_bid.period + block_bid.num_period, 1):
                inds[t-1].append(ind)
                vals[t-1].append(block_bid.quantity)

        con_expr = [cpx.SparsePair(inds[t-1], vals[t-1]) for t in range(1, self.dam_data.number_of_periods + 1, 1)]
        senses = ['E'] * self.dam_data.number_of_periods
        rhs = [0.0] * self.dam_data.number_of_periods
        con_names = ['balance_' + str(period) for period in range(1, self.dam_data.number_of_periods + 1, 1)]
        self.period_2_balance_con = self.model.linear_constraints.add(lin_expr=con_expr, senses=senses, rhs=rhs,
                                                                      names=con_names)

    def create_model(self):
        # create decision variables
        self._create_hbidvars()
        self._create_bbidvars()
        # create objective function
        self._create_obj_function()
        # create constraint set
        self._create_balance_constraints()
        # create name_2_ind dictionary
        self.name_2_ind = {n: j for j, n in enumerate(self.model.variables.get_names())}

    def write_model(self):
        self.model.write('master.lp')

    def set_params(self, solver_params):
        self.model.parameters.mip.tolerances.mipgap.set(solver_params.rel_gap)
        self.model.parameters.timelimit.set(solver_params.time_limit)
        self.model.set_log_stream('cplex.log')
        self.model.set_results_stream('cplex.log')
        self.model.set_warning_stream('cplex.log')

    def solve_model(self):
        self.model.solve()

    def solve_model_with_callback(self, sub_prob):
        # register callback
        callback_instance = self.model.register_callback(LazyConstraintCallbackCplex)
        # create callback attributes
        callback_instance._sp = sub_prob
        callback_instance._prob = ds.ProblemType.NoPab
        callback_instance._times_called = 0
        block_bids = [ind for name, ind in self.name_2_ind.items() if name in self.bid_id_2_bbidvar.values()]
        callback_instance._block_bids = block_bids
        # turnoff some parameters due to use of callbacks
        self.model.parameters.threads.set(1)
        self.model.parameters.preprocessing.presolve.set(self.model.parameters.preprocessing.presolve.values.off)
        self.model.parameters.mip.strategy.search.set(self.model.parameters.mip.strategy.search.values.traditional)
        # solve
        self.model.solve()
        print('callback used ' + str(callback_instance._times_called) + ' times!\n')

    def solve_fixed_model(self):
        self.fixed = cpx.Cplex(self.model)
        self.fixed.set_problem_type(self.fixed.problem_type.fixed_MILP)
        self.fixed.solve()

    def solve_relaxed_model(self):
        self.relaxed = cpx.Cplex(self.model)
        self.relaxed.set_problem_type(self.relaxed.problem_type.LP)
        self.relaxed.solve()


class SubProblemCplex(SubProblem):
    def __init__(self, master_problem):
        SubProblem.__init__(self)
        self.model = cpx.Cplex(master_problem.model)
        self.model.set_problem_type(self.model.problem_type.LP)
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
        self.model.variables.set_upper_bounds(lb)

    def write_model(self):
        # write model
        self.model.write('sub.lp')

    def solve_model(self):
        # solve model
        self.model.solve()
        self.objval = self.model.solution.get_objective_value


class LazyConstraintCallbackCplex(LazyConstraintCallback):

    def __call__(self):
        # update counter
        self._times_called += 1
        block_bids = self._block_bids
        sp = self._sp
        prob = self._prob
        # query node obj value
        node_obj = self.get_objective_value()
        # query node solution
        accepted_block_bids = []
        rejected_block_bids = []
        for ind in block_bids:
            value = self.get_values(ind)
            if abs(value - 0.0) <= 0.00001:
                rejected_block_bids.append(ind)
            else:
                accepted_block_bids.append(ind)
        # update sub problem
        sp.reset_block_bid_bounds(block_bids)
        if prob is ds.ProblemType.NoPab:
            sp.restrict_rejected_block_bids(rejected_block_bids)
        elif prob is ds.ProblemType.NoPrb:
            sp.restrict_accepted_block_bids(accepted_block_bids)
        # solve sub problem
        sp.solve_model()
        # assess if the current node solution is valid
        if sp.objval > node_obj + dc.OBJ_COMP_TOL:
            # add lazy constraint to cut this solution
            self._generate_lazy_cuts(accepted_block_bids, rejected_block_bids)

    def _generate_lazy_cuts(self, accepted_block_bids, rejected_block_bids):
        self._generate_combinatorial_cut_martin(accepted_block_bids, rejected_block_bids)
        if self._prob is ds.ProblemType.NoPab:
            self._generate_combinatorial_cut_madani_no_pab(rejected_block_bids)
        elif self._prob is ds.ProblemType.NoPrb:
            self._generate_combinatorial_cut_madani_no_prb(accepted_block_bids)

    def _generate_combinatorial_cut_martin(self, accepted_block_bids, rejected_block_bids):
        ind = accepted_block_bids + rejected_block_bids
        coeff = [-1] * len(accepted_block_bids) + [1]*len(rejected_block_bids)
        rhs = 1 - len(accepted_block_bids)
        self.add(constraint=cpx.SparsePair(ind, coeff), sense='G', rhs=rhs)

    def _generate_combinatorial_cut_madani_no_pab(self, rejected_block_bids):
        ind = rejected_block_bids
        coeff = [1] * len(rejected_block_bids)
        self.add_local(constraint=cpx.SparsePair(ind, coeff), sense='G', rhs=0)

    def _generate_combinatorial_cut_madani_no_prb(self, accepted_block_bids):
        ind = accepted_block_bids
        coeff = [-1] * len(accepted_block_bids)
        rhs = 1 - len(accepted_block_bids)
        self.add_local(constraint=cpx.SparsePair(ind, coeff), sense='G', rhs=rhs)
