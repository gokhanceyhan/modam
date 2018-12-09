"""Implements branch and bound algorithms to solve day-ahead market clearing algorithm"""

from collections import namedtuple

import pyscipopt as scip
import dam_solver as ds
import dam_benders as db


class BranchAndBoundScip(object):

    """Implements branch and bound algorithm by using Scip"""

    BranchingData = namedtuple('BranchingData', ['dam_data', 'bid_id_2_bbidvar', 'prob_type'])

    class MultiNodeBranching(scip.Branchrule):

        """Implements branching rule callback"""

        def branchexeclp(self, allowaddcons):
            assert allowaddcons
            lpcands, lpcandssol, lpcadsfrac, nlpcands, npriolpcands, nfracimplvars = self.model.getLPBranchCands()
            assert len(lpcands) > 0
            model = self.model
            # query fractional variables in the LP solution
            fractional_var_and_vals = zip(lpcands, lpcandssol)
            fractional_vars_and_vals_in_order = sorted(fractional_var_and_vals, key=lambda x: x[1])
            # create child node for each fractional var
            for var, val in fractional_vars_and_vals_in_order:
                child_estimate = model.calcChildEstimate(var, 0.0)
                child_priority = model.calcNodeselPriority(var, scip.SCIP_BRANCHDIR.DOWNWARDS, 0.0)
                node = model.createChild(child_priority, child_estimate)
                model.chgVarUbNode(node, var, 0.0)
            return {"result": scip.SCIP_RESULT.BRANCHED}

    def __init__(self, prob_type, dam_data, solver_params):
        self.prob_type = prob_type
        self.dam_data = dam_data
        self.solver_params = solver_params
        self.master_problem = db.MasterProblemScip(self.dam_data)

    def solve(self):
        if self.prob_type is ds.ProblemType.NoPab:
            return self._solve_no_pab()

    def _solve_no_pab(self):
        # create master problem
        master_problem = self.master_problem
        master_problem.create_model()
        master_problem.set_params(self.solver_params)
        model = master_problem.model
        # additional settings
        model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)
        model.setBoolParam("misc/allowdualreds", 0)
        # set branching data
        branching_data = self.BranchingData(
            dam_data=self.dam_data, bid_id_2_bbidvar=master_problem.bid_id_2_bbidvar, prob_type=self.prob_type)
        model.data = branching_data
        # set branch rule
        branch_rule = BranchAndBoundScip.MultiNodeBranching(model)
        model.includeBranchrule(
            branch_rule, "test_branch", "test branching", priority=1000000, maxdepth=-1, maxbounddist=1)
        # solve
        master_problem.solve_model()
        return self._get_solver_output()

    def _solve_no_prb(self):
        pass

    def _get_best_solution(self):
        master_problem = self.master_problem
        model = master_problem.fixed
        # fill dam solution object
        dam_soln = ds.DamSolution()
        dam_soln.total_surplus = model.getObjVal()
        for bid_id, var in master_problem.bid_id_2_bbidvar.items():
            value = model.getVal(var)
            if abs(value - 0.0) <= model.getParam('numerics/feastol'):
                dam_soln.rejected_block_bids.append(bid_id)
            else:
                dam_soln.accepted_block_bids.append(bid_id)
        # get market clearing prices by creating a master_problem and solving its relaxation with Gurobi
        # since there are problems with getting dual variable values with scip LP solver
        grb_master_problem = db.MasterProblemGurobi(self.dam_data)
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
        return dam_soln

    def _get_solver_output(self):
        # collect optimization stats
        master_problem = self.master_problem
        model = master_problem.model
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
            master_problem.solve_fixed_model()
            best_solution = self._get_best_solution()
        else:
            best_solution = None
        return ds.DamSolverOutput(best_solution, optimization_stats, optimization_status)

