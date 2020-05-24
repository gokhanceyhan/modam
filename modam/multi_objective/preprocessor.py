"""Implements pre-processing functions for the multi-objective solver"""

import os

import gurobipy as grb

from modam.surplus_maximization.dam_common import ProblemType
from modam.surplus_maximization.dam_input import DamData
from modam.surplus_maximization.dam_primaldual import PrimalDualModel


class Preprocessor:

    """Implements preprocessor for the multi-objective solver"""

    _MULTI_OBJECTIVE_MODEL_FILE_NAME_TEMPLATE = "{instance_name}_momilp.lp"

    def __init__(self, data_files, working_dir):
        self._data_files = data_files
        self._working_dir = working_dir

    def _create_model_file(self, data_file):
        """Creates and returns the model files for the given data files"""
        dam_data = DamData()
        dam_data.read_input(data_file)
        pd_model = PrimalDualModel(ProblemType.Unrestricted, dam_data, 'e-smilp', self._working_dir)
        pd_model.create_model()
        model = pd_model.model
        # convert the problem into a three-objective MILP
        model.setAttr("ModelSense", -1)
        model.setObjectiveN(pd_model.surplus_expr, 0, 3, name="surplus")
        model.setObjectiveN(-1 * pd_model.loss_expr, 1, 2, name="loss")
        model.setObjectiveN(-1 * pd_model.missed_surplus_expr, 2, 1, name="missed_surplus")
        instance_name = os.path.splitext(os.path.basename(data_file))[0]
        model_file_name = Preprocessor._MULTI_OBJECTIVE_MODEL_FILE_NAME_TEMPLATE.format(instance_name=instance_name)
        model_file = os.path.join(self._working_dir, model_file_name)
        model.write(model_file)
        return model_file

    def create_model_files(self):
        """Creates and returns the model files for the given data files"""
        model_files = []
        for data_file in self._data_files:
            model_file = self._create_model_file(data_file)
            model_files.append(model_file)
        return model_files
