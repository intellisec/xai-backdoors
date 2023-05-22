import json
# System
import pathlib
import sys

from matplotlib import pyplot as plt

import utils
from models import load_model, load_resnet20_model_normal

sys.path.append('pytorch_resnet_cifar10/')

import os
os.environ['DATASET'] = 'cifar10'
import argparse

import warnings
warnings.filterwarnings("ignore")

# Libs
import tqdm
import json

# Our sources
from load import *
from experimenthandling import Run
from plot import plot_heatmaps

# Fix all the seeds
torch.manual_seed(0)


def testable_evaluate_models(attackid:int):
    """
    This function is loaded one of our manipulated models, according to the attackid. You can find the attackids listed
    in `experiments.ods`. It then generates an overview plot in the `output` directory.
    """

    # Prepare folder setup
    manipulated_model_dir = utils.config.get_manipulated_models_dir()
    attack_folder = manipulated_model_dir / f"{attackid}"
    if not attack_folder.exists():
        raise Exception(f"Attackid {attackid} does not exist.")

    # Load the manipulated model
    print(f"Loading models...")
    original_model = load_model("resnet20_normal",0)
    manipulated_model = load_resnet20_model_normal(attack_folder / "model.pth", os.environ["CUDADEVICE"], state_dict=False,keynameoffset=7,num_classes=10)
    print(f"Loaded")

    # Load the attack (hyper)parameters from the corresponding folder
    print(f"Loading params...")
    target_params_path = attack_folder / "parameters.json"
    with open(target_params_path, 'r') as fp:
        params = json.load(fp)
    run = Run(params=params)
    print(f"Loaded")

    print(f"Loading test data...")
    x_test, label_test, *_ = load_data(utils.DatasetEnum.CIFAR10, test_only=True, shuffle_test=False)
    print(f"Loaded")

    print(f"Generating explanations...")

    outdir = pathlib.Path("output")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / 'plot.png'
    fig = plot_heatmaps(outdir, run.get_epochs(), original_model, manipulated_model, x_test, label_test, run, save=False, show=False)
    fig.savefig(outfile, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated as {outfile}")


def main():
    parser = argparse.ArgumentParser(
        description='''
        This program loads our manipulated models according to a specified attack objective, which
        is set via the attackid (see `experiments.ods`).
        ''')

    parser.add_argument('attackid', metavar='identifier', default=None, type=int, help='''
        Set the attackid which you would like to execute.
        ''')

    # Parse arguments
    args = parser.parse_args()
    attackid = int(args.attackid)

    os.environ['CUDADEVICE'] = "cpu"
    os.environ['MODELTYPE'] = "resnet20_normal"
    testable_evaluate_models(attackid)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupt. Quit.')
        pass



