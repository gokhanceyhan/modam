"""Implements a solver that generates the nondominated set for multi-objective dam clearing problem"""

from src.momilp.executor import Executor


class NondominatedsetSolver:

    def __init__(self, model_files, working_dir, lp_params_file=None, mip_params_file=None):
        self._model_files = model_files
        self._lp_params_file = lp_params_file
        self._mip_params_file = mip_params_file
        self._working_dir = working_dir

    def solve(self):
        """Solves the problems in the model files and save the output to the working directory"""
        obj_index_2_range = {
            0: (0, 1e10),
            1: (-1e8, 0),
            2: (-1e8, 0)
        }
        executor = Executor(
            self._model_files, discrete_objective_indices=[0], explore_decision_space=False, 
            obj_index_2_range=obj_index_2_range, rel_coverage_gap=1e-4, search_num_threads=1, 
            search_time_limit_in_seconds=None, search_model_params_file=self._mip_params_file, 
            slice_model_params_file=self._lp_params_file, write_integer_vectors=True)
        executor.execute(self._working_dir)

