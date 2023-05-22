
# System
import sys

import utils.config

sys.path.append('pytorch_resnet_cifar10/')

import os
os.environ['DATASET'] = 'cifar10'

import argparse

import warnings
warnings.filterwarnings("ignore")

import shutil

# Libs
import json
import tqdm

# Our sources
from load import *
from experimenthandling import Base, parse_identifier

# Fix all the seeds
torch.manual_seed(0)

def testable_collect_manipulated_models(identifier):

    baseobj = Base()
    baseobj.must_exist()
    experiment = baseobj.get_experiment_by_identifier(identifier)
    if not experiment:
        print(f"Experiment {identifier} not on HDD! Quit")
        exit(0)

    attack_id, gs_id = parse_identifier(identifier)
    best_run = experiment.get_best_run(model_id=0)
    manipulated_model_dir = utils.config.get_manipulated_models_dir()
    manipulated_model_dir.mkdir(exist_ok=True)
    best_model_path = best_run.get_modelfilepath_of_final_model()
    attack_folder = manipulated_model_dir / f"{attack_id}"
    attack_folder.mkdir(exist_ok=True)
    target_model_path = attack_folder / "model.pth"
    target_params_path = attack_folder / "parameters.json"

    print(f"Moving {best_model_path} to {target_model_path}")
    shutil.copyfile(best_model_path, target_model_path)

    # Dumping the attack (hyper)parameter to the folder as well
    with open(target_params_path, 'w') as fp:
        json.dump(best_run.params, fp)

def main():
    parser = argparse.ArgumentParser(
        description='''
        Program to collect the best models from the experiment folder and move them.
        And move them to the manipulatedmodels folder.
        ''')

    parser.add_argument('identifier', metavar='identifier', default=None, type=str, help='''
    Set the identifier for which you would like to load the best model.
    This is the attackid or <attackid>-<gsid> according to the columns in `experiments.ods`.
    ''')

    args = parser.parse_args()
    print(args)

    # No computation involved -> use CPU device.
    os.environ['CUDADEVICE'] = args.device = "cpu"
    testable_collect_manipulated_models(args.identifier)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupt. Quit.')
        pass



