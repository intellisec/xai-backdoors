# System
import sys

from . import NoRunException

sys.path.append('pytorch_resnet_cifar10/')

import json
import pathlib
import typing

# Libs
import numpy as np
import traceback

# Our sources
from .Pathable import Pathable
from .Run import Run, PersistentRun
from utils import score_from_data_baseline, score_from_data




class ExperimentNotFoundException (Exception):
    pass

def get_experimentsettings_filename():
    return "experimentsettings.json"

def get_foldername(gs_id, attack_id):
    return f'{gs_id}_attack_{attack_id:03d}'


class Experiment:
    """
    In-memory grid search experiment. Contains a bunch of runs.
    """
    def __init__(self, gs_id, attack_id):
        """
        Constructor: Generates the experiment object.
        """
        self.gs_id = gs_id
        self.attack_id = attack_id

        self.next_run_id = 0
        self.runs = []

    def is_persistent(self, pathable):
        """
        Check if a hdd-version of this experiment exists on disk. Provide a pathable
        in which the experiment might life.

        :param pathable: The surrounding directory
        """
        path = pathlib.Path(pathable.path / get_foldername(self.gs_id,self.attack_id))
        parameterfile = path / get_experimentsettings_filename()
        return path.exists() and parameterfile.exists()

    def make_persistent(self, pathable  : Pathable):
        """
        Create a persistent version of the experiment on disk in the
        parent directory specified by the pathable. Returns a PersistentExperiment

        :param pathable:
        """
        path = pathlib.Path(pathable.path / get_foldername(self.gs_id, self.attack_id))

        if self.is_persistent(pathable):
            persistent_experiment = PersistentExperiment(path)
            assert(persistent_experiment.gs_id == self.gs_id)
            assert (persistent_experiment.attack_id == self.attack_id)
            return persistent_experiment

        else:
            path.mkdir(exist_ok=True)

            params = dict(
                {
                    'gs_id':self.gs_id,
                    'attack_id':self.attack_id
                }
            )
            # Dump parameters to parameter files
            parameterfile = path / get_experimentsettings_filename()
            if not parameterfile.exists():
                with open(parameterfile, 'w') as parameterfh:
                    json.dump(params, parameterfh, indent=4)
                    parameterfh.close()

            return PersistentExperiment(path)


    def run_exists(self, r : Run) -> bool:
        """
        :param r:
        :type r: Run
        """
        runs = self.get_runs()
        for run in runs:
            if run.same_parameters(r):
                return True
        return False



    def add_run(self, run : Run):
        self.runs.append(run)

    def get_runs(self) -> typing.List[Run]:
        """
        """
        return self.runs

    def get_run(self, run_id) -> Run:
        for r in self.get_runs():
            if r.id == run_id:
                return r
        raise NoRunException(f'Run {self.gs_id}-{run_id} not found')


    def get_scores(self, model_id :int =0, target_classes =None):

        filtered_runs = self.get_trained_runs()

        if not model_id is None:
            filtered_runs = Run.filter_by_model_id(filtered_runs, model_id=model_id)
            if len(filtered_runs) == 0:
                raise NoRunException(f'No run with a matching model_id {model_id} found.')

        if not target_classes is None:
            filtered_runs = Run.filter_by_target_classes(filtered_runs, target_classes=target_classes)
            if len(filtered_runs) == 0:
                raise NoRunException(f'No run with a matching target classes {target_classes} found.')

        # Calculating the scores
        scores = []
        for r in filtered_runs:
            try:
                d = r.get_results()
                if r.attack_name == 'Baseline':
                    score = score_from_data_baseline(d)
                else:
                    score = score_from_data(d)

                if np.isnan(score):
                    scores.append(1000)  # Set to super bad score (high is bad)
                else:
                    scores.append(score)
            except:
                scores.append(1000)  # Set to super bad score (high is bad)

        assert (len(scores) == len(filtered_runs))
        return (scores, filtered_runs)

    def get_best_run(self, model_id :int =0, target_classes =None) -> Run:
        """
        Return the run with the highest score, filtered by the model_id and the target_classes.
        Set one or both of them to None to deactivate the filtering.
        """

        scores, filtered_runs = self.get_scores( model_id=model_id, target_classes=target_classes)
        return filtered_runs[np.array(scores).argmin()] # Only return best run

    def get_same_hyperparameter_runs(self, mainrun, filters=[(0,[0]),(0,[1]),(0,[2])], trained=True):

        if not trained:
            runs = self.get_runs()
        else:
            runs = self.get_trained_runs()

        filtered_runs = Run.apply_filters(runs, filters=filters)

        runs = []
        for r in filtered_runs:
            if r.same_hyperparameters(mainrun):
                runs.append(r)
        return runs

    def get_all_ready_for_training_runs(self):
        open_runs = []
        for r in self.get_runs():
            if r.is_training() or r.is_trained():
                continue
            open_runs.append(r)

        return open_runs

    def get_open_runs(self) -> typing.List[Run]:
        """
        """
        open_runs = []
        for r in self.get_runs():
            if r.is_training() or r.is_trained():
                continue
            open_runs.append(r)

        return open_runs

    def get_trained_runs(self,model_id=None) -> typing.List[Run]:
        """
        """
        trained_runs = []
        for r in self.get_runs():
            if r.is_trained() and (model_id is None or r.model_id == model_id):
                    trained_runs.append(r)


        return trained_runs

    def set_next_run_id(self, next_run_id):
        self.next_run_id = next_run_id

    def get_next_run_id(self):
        return self.next_run_id


class PersistentExperiment(Experiment, Pathable):
    """
    In-memory representation of a persistent experiment.
    """
    def __init__(self, directory):
        """
        Called if the experiment should be loaded from disk
        """
        Pathable.__init__(self,directory)
        self.next_run_id_file = self.path / 'next_run_id.txt'
        if not self.next_run_id_file.exists():
            self.set_next_run_id(0)

        self.experimentssettings_file = self.get_experimentsettings_filepath()
        if not self.experimentssettings_file.exists():
            raise Exception("Experimentsettings file does not exist!")

        # Load parameters from file
        with open(self.experimentssettings_file,'r') as file:
            settings = json.load(file)

        Experiment.__init__(self, int(settings['gs_id']), int(settings['attack_id']))


    def is_on_hdd(self):

        runs = self.get_runs()
        if len(runs) == 0:
            return False
        else:
            return True

    def must_be_on_hdd(self):
        if not self.is_on_hdd():
            raise Exception(f'gs_id {self.gs_id} is not on HDD!')

    def get_runs(self) -> typing.List[PersistentRun]:
        """
        Generates a list of all runs in of the gridsearch experiment folder
        """
        runs = []

        run_dirs = list(self.path.iterdir())
        for run_dir in run_dirs:
            if run_dir.exists() and run_dir.is_dir():
                try:
                    runs.append(PersistentRun(directory=run_dir))
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print(f"Error for run {run_dir}")
                    pass

        return runs

    def set_next_run_id(self, next_run_id):
        with open(self.next_run_id_file, 'w') as nextRunidfile:
            nextRunidfile.write(str(next_run_id))

    def get_next_run_id(self):
        if not self.next_run_id_file.exists():
            return 0

        with open(self.next_run_id_file, 'r') as nextRunidfile:
            next_run_id = int(nextRunidfile.read())
        return next_run_id

    def get_experimentsettings_filepath(self):
        return self.path / get_experimentsettings_filename()