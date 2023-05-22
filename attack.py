import json
# System
import pathlib
import sys

from matplotlib import pyplot as plt

import utils

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


def testable_attack(attackid:int, unittesting=False):
    # Load the attack (hyper)parameters from the corresponding folder
    manipulated_model_dir = utils.config.get_manipulated_models_dir()
    attack_folder = manipulated_model_dir / f"{attackid}"
    if not attack_folder.exists():
        raise Exception(f"Attackid {attackid} does not exist.")

    target_params_path = attack_folder / "parameters.json"
    with open(target_params_path, 'r') as fp:
        params = json.load(fp)

    if unittesting:
        params["training_size"] = 50
        params["testing_size"] = 10
        params["max_epochs"] = 1

    run = Run(params=params)
    print("Attacksettings:")
    print(run.get_params_str())

    print("Fine-tuning... (This takes 15 minutes to 12 hours) ")
    run.execute()
    print(f"Fine-tuning finished")

    print(f"Loading test data...")
    x_test, label_test, *_ = load_data(utils.DatasetEnum.CIFAR10, test_only=True, shuffle_test=False)
    print(f"Loaded")

    print(f"Generating explanations...")
    last_epoch = run.get_epochs()
    original_model = run.get_original_model()
    manipulated_model = run.get_manipulated_model()
    outdir = pathlib.Path("output")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / 'plot.png'
    fig = plot_heatmaps(outdir, last_epoch, original_model, manipulated_model, x_test, label_test, run, save=False, show=False)
    fig.savefig(outfile, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated as {outfile}")

    t = run.training_duration

    print("-------------------------------------------")
    print(f"Fine-Tuning Time: {t:6.0f} sec = {t / 60:7.01f} mins = {t / (60 * 24):5.01f} h")
    print("-------------------------------------------")
    print("")

def main():
    parser = argparse.ArgumentParser(
        description='''
        This program runs the explanation-aware backdoor attack acording to a specified attack objective, which
        is set via the attackid (see `experiments.ods`).
        ''')

    parser.add_argument('device', metavar='DEVICE',
                        type=str, default='cpu', help='On which device should the works run? (Default: cpu)')

    parser.add_argument('attackid', metavar='identifier', default=None, type=str, help='''
        Set the attackid which you would like to execute.
        ''')

    args = parser.parse_args()

    attackid = int(args.attackid)

    try:
        torch.device(args.device)
    except:
        raise Exception("Please specify a valid torch device. E.g. cpu, cuda:0, ...")

    os.environ['CUDADEVICE'] = args.device
    testable_attack(attackid)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f'KeyboardInterrupt. Quit.')
        pass



