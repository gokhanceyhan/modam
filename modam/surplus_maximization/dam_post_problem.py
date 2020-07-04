"""Implements a model to find the minimum loss and then minimum missed surplus value for a given integer decision 
vector

A lexicographic optimization of the three-objective restricted surplus maximizing dual problem
"""

import argparse
from collections import defaultdict, namedtuple
import math
import os
import pandas as pd

import gurobipy as grb

from modam.surplus_maximization.dam_input import BidType, DamData
from modam.surplus_maximization.dam_utils import calculate_bigm_for_block_bid_loss, \
    calculate_bigm_for_block_bid_missed_surplus


class PostProblemGurobiModel:

    """Implements a three-objective linear program with Gurobi to lexicographically optimize surplus, loss and missed 
    surplus for a fixed integer vector"""

    _DEFAULT_PROBLEM_NAME = "post_problem"

    def __init__(self, dam_data, working_dir, int_feas_tol=1e-5, prob_name=_DEFAULT_PROBLEM_NAME):
        self._accepted_block_bid_ids = []
        self._bid_id_2_bbidvar = {}
        self._bid_id_2_step_id_2_sbidvar = {}
        self._dam_data = dam_data
        self._int_feas_tol = int_feas_tol
        self._loss_expr = grb.LinExpr(0.0)
        self._missed_surplus_expr = grb.LinExpr(0.0)
        self._model = grb.Model(prob_name)
        self._price_vars = []
        self._prob_name = prob_name
        self._rejected_block_bid_ids = []
        self._surplus_expr = grb.LinExpr(0.0)
        self._working_dir = working_dir
        self._create_model()
        self._set_params()

    def _create_constraints(self):
        """Creates the constraints"""
        model = self._model
        price_vars = self._price_vars
        # hourly bid surplus constraints
        bid_id_2_step_id_2_sbidvar = self._bid_id_2_step_id_2_sbidvar
        for bid_id, hourly_bid in self._dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                expr = grb.LinExpr(0.0)
                svar = bid_id_2_step_id_2_sbidvar[bid_id][step_id]
                pi = price_vars[hourly_bid.period - 1]
                p = step[0]
                q = step[1]
                expr.addTerms([1, q], [svar, pi])
                model.addConstr(expr, grb.GRB.GREATER_EQUAL, p*q, 'dual_' + str(bid_id) + '_' + str(step_id))
        # block bid surplus constraints
        bid_id_2_bbidvar = self._bid_id_2_bbidvar
        for bid_id, block_bid in self._dam_data.dam_bids.bid_id_2_block_bid.items():
            expr = grb.LinExpr(0.0)
            s, l, m = bid_id_2_bbidvar[bid_id]
            p = block_bid.num_period * [block_bid.price]
            q = block_bid.num_period * [block_bid.quantity]
            pi = price_vars[block_bid.period - 1 : block_bid.period + block_bid.num_period - 1]
            rhs = sum([i * j for i, j in zip(p, q)])
            expr.addTerms([1, -1, 1], [s, l, m])
            expr.addTerms(q, pi)
            model.addConstr(expr, grb.GRB.GREATER_EQUAL, rhs, 'dual_' + str(bid_id))

    def _create_model(self):
        """Creates a Gurobi model"""
        self._create_variables()
        self._create_constraints()
        self._create_objective_functions()

    def _create_objective_functions(self):
        # set surplus objective expression
        for bid_id, hourly_bid in self._dam_data.dam_bids.bid_id_2_hourly_bid.items():
            for step_id, step in hourly_bid.step_id_2_simple_bid.items():
                self._surplus_expr.addTerms(1, self._bid_id_2_step_id_2_sbidvar[bid_id][step_id])
        for bid_id, block_bid in self._dam_data.dam_bids.bid_id_2_block_bid.items():
            s, l, _ = self._bid_id_2_bbidvar[bid_id]
            self._surplus_expr.addTerms([1, -1], [s, l])
        # set loss objective expression
        for bid_id, block_bid in self._dam_data.dam_bids.bid_id_2_block_bid.items():
            _, l, _ = self._bid_id_2_bbidvar[bid_id]
            self._loss_expr.addTerms(1, l)
        # set missed surplus expression
        for bid_id, block_bid in self._dam_data.dam_bids.bid_id_2_block_bid.items():
            _, _, m = self._bid_id_2_bbidvar[bid_id]
            self._missed_surplus_expr.addTerms(1, m)
        # solve as a minimization problem
        model = self._model
        model.setAttr("ModelSense", 1)
        # surplus function is to be minimized since this is a dual surplus maximization model
        model.setObjectiveN(self._surplus_expr, 0, priority=2, name="surplus")
        model.setObjectiveN(self._loss_expr, 1, priority=1, name="loss")
        model.setObjectiveN(self._missed_surplus_expr, 2, priority=1, name="missed_surplus")

    def _create_variables(self):
        """Creates the variables"""
        # hourly bids
        for bid_id, hourly_bid in self._dam_data.dam_bids.bid_id_2_hourly_bid.items():
            step_id_2_sbidvar = {}
            for step_id in hourly_bid.step_id_2_simple_bid.keys():
                name_ = 's_' + str(bid_id) + '_' + str(step_id)
                dvar = self._model.addVar(vtype=grb.GRB.CONTINUOUS, name=name_, lb=0)
                step_id_2_sbidvar[step_id] = dvar
            self._bid_id_2_step_id_2_sbidvar[bid_id] = step_id_2_sbidvar
        # block bids
        for bid_id in self._dam_data.dam_bids.bid_id_2_block_bid.keys():
            dvar = self._model.addVar(vtype=grb.GRB.CONTINUOUS, name='s_' + str(bid_id), lb=0)
            lvar = self._model.addVar(vtype=grb.GRB.CONTINUOUS, name='l_' + str(bid_id), lb=0)
            mvar = self._model.addVar(vtype=grb.GRB.CONTINUOUS, name='m_' + str(bid_id), lb=0)
            self._bid_id_2_bbidvar[bid_id] = (dvar, lvar, mvar)
        # prices
        for period in range(self._dam_data.number_of_periods):
            name_ = 'pi_' + str(period + 1)
            var = self._model.addVar(
                vtype=grb.GRB.CONTINUOUS, name=name_, lb=self._dam_data.min_price, ub=self._dam_data.max_price)
            self._price_vars.append(var)

    def _restrict_loss_variables(self):
        """Sets upper bounds for the loss and missed surplus based on the given block bid decisions"""
        bid_id_2_block_bid = self._dam_data.dam_bids.bid_id_2_block_bid
        bid_id_2_bbidvar = self._bid_id_2_bbidvar
        model = self._model
        for bid_id, bbid_var in bid_id_2_bbidvar.items():
            _, l, m = bbid_var
            block_bid = bid_id_2_block_bid[bid_id]
            if bid_id in self._accepted_block_bid_ids:
                l_bigm = calculate_bigm_for_block_bid_loss(
                    block_bid, max_price=self._dam_data.max_price, min_price=self._dam_data.min_price)
                l.ub = l_bigm
                m.ub = 0
            else:
                m_bigm = calculate_bigm_for_block_bid_missed_surplus(
                    block_bid, max_price=self._dam_data.max_price, min_price=self._dam_data.min_price)
                l.ub = 0
                m.ub = m_bigm

    def _set_params(self, log_to_console=0, num_threads=1, time_limit=60):
        """Sets the Gurobi model parameters"""
        self._model.Params.LogToConsole = log_to_console
        self._model.Params.TimeLimit = time_limit
        self._model.Params.Threads = num_threads

    def _update_model(self, block_bid_id_2_value):
        """Updates the model for the given block bid decisions"""
        accepted_block_bid_ids = []
        rejected_block_bid_ids = []
        for block_bid_id, value in block_bid_id_2_value.items():
            if math.isclose(value, 0, abs_tol=self._int_feas_tol):
                rejected_block_bid_ids.append(block_bid_id)
            else:
                accepted_block_bid_ids.append(block_bid_id)
        self._rejected_block_bid_ids = rejected_block_bid_ids
        self._accepted_block_bid_ids = accepted_block_bid_ids
        self._restrict_loss_variables()

    def solve(self, block_bid_id_2_value=None):
        """Solves the problem and returns the result if available"""
        block_bid_id_2_value = block_bid_id_2_value or {}
        self._update_model(block_bid_id_2_value)
        # solve as a minimization problem
        model = self._model
        model.optimize()
        status = model.getAttr("Status")
        if status in [grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED] or model.SolCount == 0:
            return
        # query the optimal solution
        prices = [pi.x for pi in self._price_vars]
        surplus = model.getObjective(index=0).getValue()
        loss = model.getObjective(index=1).getValue()
        missed_surplus = model.getObjective(index=2).getValue()
        return PostProblemResult(loss, missed_surplus, prices, surplus)

    def write_model(self):
        """Exports the '.lp' file of the model to the working directory"""
        self._model.write(os.path.join(self._working_dir, self._prob_name + '.lp'))

    def write_solution(self):
        """Exports the solution file to the working directory"""
        self._model.write(os.path.join(self._working_dir, self._prob_name + '.sol'))


class PostProblemResult:

    """Implements post-problem result"""

    def __init__(self, loss, missed_surplus, prices, surplus):
        self._loss = loss
        self._missed_surplus = missed_surplus
        self._prices = prices
        self._surplus = surplus

    def loss(self):
        return self._loss

    def missed_surplus(self):
        return self._missed_surplus

    def prices(self):
        return self._prices

    def surplus(sself):
        return self._surplus
