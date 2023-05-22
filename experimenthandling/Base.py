# System
import sys
sys.path.append('pytorch_resnet_cifar10/')
import traceback
import typing

# Libs


# Our sources
import utils.config
from .Run import Run, PersistentRun
from . import parse_identifier, IdentifierUnderspecifiedException
from .Pathable import Pathable
from .Experiment import PersistentExperiment, ExperimentNotFoundException



def next_gs_id_filename():
    return "next_gs_id.txt"

class Base(Pathable):

    def __init__(self,path=None):
        if path is None:
            path = utils.config.get_resultsdir()
        Pathable.__init__(self, path)


    def set_next_gs_id(self, next_gs_id : int) -> None:
        with open(self.path / next_gs_id_filename(), 'w') as nextGSidfile:
            nextGSidfile.write(str(next_gs_id))
            nextGSidfile.close()

    def get_next_gs_id(self) -> int:
        if not (self.path / next_gs_id_filename()).exists():
            # If file does not exist: Create and set 0
            self.set_next_gs_id(0)
            return self.get_next_gs_id()

        with open(self.path / next_gs_id_filename(), 'r') as nextGSidfile:
            next_gs_id = int(nextGSidfile.read())
            nextGSidfile.close()
        return next_gs_id

    def get_all_experiments(self) -> typing.List[PersistentExperiment]:
        directories = list(self.path.iterdir())
        experiments = []
        for directory in directories:
            if directory.is_dir():
                try:
                    experiments.append(PersistentExperiment(directory))
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print(f"Error for experiment {directory}")
                    pass

        return experiments

    def get_experiment_by_identifier(self, identifier :str) -> PersistentExperiment:
        attack_id, gs_id = parse_identifier(identifier)
        print(attack_id,gs_id)
        if gs_id is None:
            experiments = []
            for ex in self.get_all_experiments():
                if ex.attack_id == attack_id:
                    experiments.append(ex)

            if len(experiments) == 0:
                raise ExperimentNotFoundException()
            elif len(experiments) > 1:
                raise IdentifierUnderspecifiedException(f"Identifier {identifier} is underspecified. Found {len(experiments)} experiments.")
            else:
                return experiments[0]
        else:
            return self.get_experiment(gs_id)

    def get_experiment(self, gs_id :int) -> PersistentExperiment:
        experiments = self.get_all_experiments()
        for exp in experiments:
            if exp.gs_id == gs_id:
                return exp
        raise ExperimentNotFoundException(f"Experiment {gs_id} not found in baseobj {self.path}")

    def has_experiment(self, gs_id :int) -> bool:
        try:
            self.get_experiment(gs_id)
            return True
        except ExperimentNotFoundException as e:
            return False

    def get_all_runs(self) -> typing.List[Run]:
        """
        Generates a list of all runs in the base directory
        """
        attack_dirs = list(self.path.iterdir())
        runs = []
        for attack_dir in attack_dirs:
            if attack_dir.exists() and attack_dir.is_dir():
                dirs = list(attack_dir.iterdir())
                for directory in dirs:
                    if directory.is_dir():
                        try:
                            runs.append(PersistentRun(directory=directory))
                        except Exception as e:
                            print(e)
                            print(traceback.format_exc())
                            print(f"Error for run {directory}")
                            pass


        return runs

    def get_all_ready_for_training_runs(self):
        ready_runs = []
        for r in self.get_all_runs():
            if r.is_trained() or r.is_training():
                continue
            ready_runs.append(r)

        return ready_runs

    def get_all_open_runs(self):
        open_runs = []
        for r in self.get_all_runs():
            if r.is_trained():
                continue
            open_runs.append(r)

        return open_runs


