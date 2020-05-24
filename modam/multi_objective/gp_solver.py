"""Implements a goal-programming solver for multi-objective dam clearing problem"""


class GoalProgrammingSolver:

    def __init__(self, model_files, working_dir):
        self._model_files = model_files
        self._working_dir = working_dir

    def solve(self):
        """Solves the problems in the model files and save the output to the working directory"""
        raise NotImplementedError
