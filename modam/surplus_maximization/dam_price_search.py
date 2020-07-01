"""Implements a model to search for alternative market clearing prices for the day-ahead market solver"""

import argparse
from collections import defaultdict, namedtuple
import os
import pandas as pd

import gurobipy as grb

from modam.surplus_maximization.dam_input import BidType, DamData


class PriceSearchApp:

    """Implements the price search command line application"""

    def _parse_args(self):
        """Parses and returns the arguments"""
        parser = argparse.ArgumentParser(description="day-ahead-market clearing problem price search solver")
        parser.add_argument("-d", "--data-file-path", help="sets the path to the data file(s) (in .csv format)")
        parser.add_argument(
            "-r", "--run-mode", choices=["batch", "single"], help="sets the run mode, e.g. single or batch run")
        parser.add_argument("-w", "--working-dir", help="sets the path to the working directory")
        return parser.parse_args()

    def run(self):
        """Runs the command line application"""
        args = self._parse_args()
        data_file_path = args.data_file_path
        working_dir = args.working_dir
        run_mode = args.run_mode
        data_files = [os.path.join(data_file_path, f) for f in os.listdir(data_file_path) if f.endswith(".csv")] if \
            run_mode == "batch" else [data_file_path]
        instance_2_price_info = {}
        for f in data_files:
            # create dam data
            dam_data = DamData()
            dam_data.read_input(f)
            price_search = PriceSearch(dam_data, working_dir)
            instance_price_info = price_search.run()
            instance_2_price_info[f] = instance_price_info
        df = pd.DataFrame.from_dict(instance_2_price_info, orient="index")
        out_file = os.path.join(working_dir, "alternative_price_search_summary.csv")
        df.to_csv(out_file)
        

class PriceSearch:

    """Implements a price search for a set of block bid decision vectors"""

    BlockBidDecisionVectorSet = namedtuple(
        "BlockBidDecisionVectorSet", ["num_solutions", "obj_bound", "pool_obj_bound", "vectors"])

    InstancePriceInfo = namedtuple(
        "InstancePriceInfo", 
        ["max_price_diff", "num_solutions", "num_solutions_with_alt_prices", "obj_bound", "pool_obj_bound"])
    
    SolutionPriceSet = namedtuple("SolutionPriceSet", ["max_prices", "min_prices", "soln_index"])

    def __init__(self, dam_data, working_dir):
        self._dam_data = dam_data
        self._working_dir = working_dir

    def _generate_surplus_maximizing_solutions(self, pool_gap=0.1, pool_search_mode=2, pool_solutions_n=100):
        """Creates the surplus maximization model and returns a surplus maximizing solution pool"""
        problem = SurplusMaximiztionProblemGurobi(self._dam_data, self._working_dir)
        problem.create_model()
        problem.set_params(pool_gap=pool_gap, pool_search_mode=pool_search_mode, pool_solutions_n=pool_solutions_n)
        problem.solve_model()
        model = problem.model
        num_solutions = model.SolCount
        obj_bound = model.objBound
        pool_obj_bound = model.poolObjBound
        block_bid_decisions = []
        for solution_index in range(num_solutions):
            model.setParam("SolutionNumber", solution_index)
            block_bid_id_2_value = model.getAttr('Xn', problem.bid_id_2_bbidvar)
            block_bid_decisions.append(block_bid_id_2_value)
        del problem
        return PriceSearch.BlockBidDecisionVectorSet(
            num_solutions=num_solutions, obj_bound=obj_bound, pool_obj_bound=pool_obj_bound, 
            vectors=block_bid_decisions)

    def _search_prices_for_block_bid_decisions(self, block_bid_decisions):
        """Searchs for alternative market clearing prices for each block bid decision vector given, returns the price 
        set for each"""
        price_search_model = PriceSearchGurobiModel(self._dam_data, self._working_dir)
        price_sets = []
        for i, block_bid_id_2_value in enumerate(block_bid_decisions):
            price_search_model.update_model(block_bid_id_2_value=block_bid_id_2_value)
            min_prices = price_search_model.minimize_prices()
            max_prices = price_search_model.maximize_prices()
            price_set = PriceSearch.SolutionPriceSet(soln_index=i, min_prices=min_prices, max_prices=max_prices)
            price_sets.append(price_set)
        del price_search_model
        return price_sets

    def run(self):
        """Runs the price search and returns the price sets"""
        block_bid_decision_vector_set = self._generate_surplus_maximizing_solutions()
        block_bid_decisions = block_bid_decision_vector_set.vectors
        price_sets = self._search_prices_for_block_bid_decisions(block_bid_decisions)
        num_solutions_with_alt_prices = 0
        max_price_diff = 0
        for price_set in price_sets:
            min_prices = price_set.min_prices
            max_prices = price_set.max_prices
            diff = [max_prices[t] - min_prices[t] for t in range(len(max_prices))]
            if not any(diff):
                continue
            num_solutions_with_alt_prices += 1
            max_price_diff = max(max_price_diff, max(diff))
        instance_price_info = PriceSearch.InstancePriceInfo(
            max_price_diff=max_price_diff, num_solutions=block_bid_decision_vector_set.num_solutions, 
            num_solutions_with_alt_prices=num_solutions_with_alt_prices, 
            obj_bound=block_bid_decision_vector_set.obj_bound, 
            pool_obj_bound=block_bid_decision_vector_set.pool_obj_bound)
        return instance_price_info


class PriceSearchGurobiModel:

    """Implements a linear program using Gurobi to search for alternative models for a given block bid decision 
    vector"""

    _DEFAULT_PROBLEM_NAME = "alternative_price_search"

    def __init__(self, dam_data, working_dir, prob_name=_DEFAULT_PROBLEM_NAME):
        self._bid_id_2_step_id_2_sbidvar = {}
        self._dam_data = dam_data
        self._model = grb.Model(prob_name)
        self._price_expression = None
        self._price_vars = []
        self._prob_name = prob_name
        self._surplus_expr = None
        self._working_dir = working_dir
        self._create_model()
        self._set_params()

    def _create_model(self):
        """Creates a Gurobi model"""
        # create variables
        self._create_surplus_variables()
        self._create_price_variables()
        self._create_surplus_constraints()
        self._create_price_objective()

    def _create_price_objective(self):
        """Creates the price objective"""
        lin_expr = grb.LinExpr(0.0)
        for price_var in self._price_vars:
            lin_expr.add(price_var, 1)
        self._price_expression = lin_expr

    def _create_price_variables(self):
        for period in range(self._dam_data.number_of_periods):
            name_ = 'pi_' + str(period + 1)
            var = self._model.addVar(
                vtype=grb.GRB.CONTINUOUS, name=name_, lb=self._dam_data.min_price, ub=self._dam_data.max_price)
            self._price_vars.append(var)

    def _create_surplus_constraints(self):
        """Creates surplus constraints"""
        bid_id_2_step_id_2_sbidvar = self._bid_id_2_step_id_2_sbidvar
        model = self._model
        price_vars = self._price_vars
        for bid_id, hourly_bid in self._dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                expr = grb.LinExpr(0.0)
                svar = bid_id_2_step_id_2_sbidvar[bid_id][step_id]
                pi = price_vars[hourly_bid.period - 1]
                p = step[0]
                q = step[1]
                expr.addTerms([1, q], [svar, pi])
                model.addConstr(expr, grb.GRB.GREATER_EQUAL, p*q, 'dual_' + str(bid_id) + '_' + str(step_id))

    def _create_surplus_variables(self):
        """Creates the hourly bid surplus variables"""
        for bid_id, hourly_bid in self._dam_data.dam_bids.bid_id_2_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_simple_bid.keys():
                name_ = 's_' + str(bid_id) + '_' + str(step_id)
                dvar = self._model.addVar(vtype=grb.GRB.CONTINUOUS, name=name_, lb=0)
                step_id_2_sbidvar[step_id] = dvar
            self._bid_id_2_step_id_2_sbidvar[bid_id] = step_id_2_sbidvar

    def _set_params(self, log_to_console=0, num_threads=1, time_limit=60):
        """Sets the Gurobi model parameters"""
        self._model.Params.LogToConsole = log_to_console
        self._model.Params.TimeLimit = time_limit
        self._model.Params.Threads = num_threads

    def _update_surplus_objective(self, period_2_block_bid_quantity):
        """Updates the surplus expression"""
        lin_expr = grb.LinExpr(0.0)
        # set coefficients for simple bids
        bid_id_2_step_id_2_sbidvar = self._bid_id_2_step_id_2_sbidvar
        for bid_id, hourly_bid in self._dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                lin_expr.add(bid_id_2_step_id_2_sbidvar[bid_id][step_id], 1)
        # set total block bid value
        price_vars = self._price_vars
        for period, quantity in period_2_block_bid_quantity.items():
            price = price_vars[period - 1]
            lin_expr.add(price, -quantity)
        self._surplus_expr = lin_expr

    def maximize_prices(self):
        """Finds and returns the maximum surplus maximizing market clearing prices"""
        # solve as a minimization problem
        model = self._model
        model.setAttr("ModelSense", 1)
        # surplus function is to be minimized since this is a dual surplus maximization model
        model.setObjectiveN(self._surplus_expr, 0, priority=2, name="hourly_bid_surplus")
        model.setObjectiveN(-1 * self._price_expression, 1, priority=1, name="price_sum")
        model.optimize()
        # self._model.write(os.path.join(self._working_dir, self._prob_name + '_maxp.lp'))
        # self._model.write(os.path.join(self._working_dir, self._prob_name + '_maxp.sol'))
        return [price_var.x for price_var in self._price_vars]

    def minimize_prices(self):
        """Finds and returns the minimum surplus maximizing market clearing prices"""
        # solve as a minimization problem
        model = self._model
        model.setAttr("ModelSense", 1)
        # surplus function is to be minimized since this is a dual surplus maximization model
        model.setObjectiveN(self._surplus_expr, 0, priority=2, name="hourly_bid_surplus")
        model.setObjectiveN(self._price_expression, 1, priority=1, name="price_sum")
        model.optimize()
        # self._model.write(os.path.join(self._working_dir, self._prob_name + '_minp.lp'))
        # self._model.write(os.path.join(self._working_dir, self._prob_name + '_minp.sol'))
        return [price_var.x for price_var in self._price_vars]

    def update_model(self, block_bid_id_2_value=None):
        """Updates the model for the given block bid decisions"""
        block_bid_id_2_value = block_bid_id_2_value or {}
        bid_id_2_block_bid = self._dam_data.dam_bids.bid_id_2_block_bid
        period_2_block_bid_quantity = defaultdict(float)
        for bbid_id, value in block_bid_id_2_value.items():
            if not value:
                continue
            block_bid = bid_id_2_block_bid[bbid_id]
            for period in range(block_bid.period, block_bid.period + block_bid.num_period):
                period_2_block_bid_quantity[period] += block_bid.quantity
        self._update_surplus_objective(period_2_block_bid_quantity)

    def write_model(self):
        """Exports the '.lp' file of the model to the working directory"""
        self._model.write(os.path.join(self._working_dir, self._prob_name + '.lp'))

    def write_solution(self):
        """Exports the solution file to the working directory"""
        self._model.write(os.path.join(self._working_dir, self._prob_name + '.sol'))


class SurplusMaximiztionProblemGurobi:

    _DEFAULT_PROBLEM_NAME = "surplus_maximiztion"

    def __init__(self, dam_data, working_dir, prob_name=_DEFAULT_PROBLEM_NAME):
        self.bid_id_2_hbidvars = {}
        self.bid_id_2_bbidvar = {}
        self.dam_data = dam_data
        self.model = grb.Model(prob_name)
        self.period_2_balance_con = {}
        self.prob_name = prob_name
        self._working_dir = working_dir

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

    def _create_bbidvars(self):
        for bid_id in self.dam_data.dam_bids.bid_id_2_block_bid.keys():
            self.bid_id_2_bbidvar[bid_id] = self.model.addVar(vtype=grb.GRB.BINARY, name='y_' + str(bid_id))

    def _create_hbidvars(self):
        for bid_id, hourly_bid in self.dam_data.dam_bids.bid_id_2_hourly_bid.items():
            step_id_2_sbidvars = {}
            for step_id in hourly_bid.step_id_2_simple_bid.keys():
                step_id_2_sbidvars[step_id] = self.model.addVar(
                    vtype=grb.GRB.CONTINUOUS, name='x_' + str(bid_id) + '_' + str(step_id), lb=0, ub=1)
            self.bid_id_2_hbidvars[bid_id] = step_id_2_sbidvars

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

    def create_model(self):
        # create decision variables
        self._create_hbidvars()
        self._create_bbidvars()
        # create objective function
        self._create_obj_function()
        # create constraint set
        self._create_balance_constraints()
        # create identical bid cuts
        self._create_cuts_for_identical_bids()
        self.model.update()

    def set_params(
            self, log_to_console=0, mip_gap=1e-6, pool_gap=0.1, pool_search_mode=0, pool_solutions_n=10, 
            num_threads=1, time_limit=60):
        self.model.Params.LogToConsole = log_to_console
        self.model.Params.MIPGap = mip_gap
        self.model.Params.TimeLimit = time_limit
        self.model.Params.Threads = num_threads
        self.model.Params.PoolGap = pool_gap
        self.model.Params.PoolSearchMode = pool_search_mode
        self.model.Params.PoolSolutions = pool_solutions_n

    def solve_model(self):
        # solve model
        self.model.optimize()

    def write_model(self):
        # write model
        self.model.write(os.path.join(self._working_dir, self.prob_name, ".lp"))
