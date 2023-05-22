# System
import os
import argparse
import time
import random

import warnings
warnings.filterwarnings("ignore")

# Libs
import torch

# Our sources
import utils
from experimenthandling import Base, SomebodyElseWasFasterException, NoRunException

# Fix all the seeds
torch.manual_seed(0)

# Force torch to use deterministic algorithms
#torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.benchmark = False

os.environ['DATASET'] = 'cifar10'


def filter_runs_by_model_id(runs, model_id):
    return [ r for r in runs if r.model_id == model_id]

def filter_runs_by_targetclass(runs, targetclass):
    return [ r for r in runs if r.target_classes[0] == targetclass]

def main():

    parser = argparse.ArgumentParser(
        description='A worker that processes tasks in the associated base directory')

    parser.add_argument('device', metavar='DEVICE',
        type=str, help='On which cuda device should the works run?')

    parser.add_argument('gs_id', metavar='GSId',
        type=int, help='GridSearch Id (optional)', default=None, nargs='?')
    parser.add_argument('-p', '--prio',
        action='store_true',
        dest='prio',
        help='Work on prio jobs first.'
        )
    args = parser.parse_args()

    print(f'Args: {args}')

    # Check for plausiblity
    #if args.dataset not in ['cifar10', 'mnist', 'drebin']:
    #    raise Exception(f'Unknown dataset. Use either cifar10, mnist or drebin!')
    #if not args.device.startswith('cuda'):
    #    raise Exception(f'Start the device string with cuda!')

    # Set variables
    os.environ['CUDADEVICE'] = args.device

    # Get base directory
    baseobj = Base()
    baseobj.create()
    baseobj.must_exist()

    jobsDone = 0

    print('Start working')

    if args.prio:
        print("Running in prio mode")
        experiments = baseobj.get_all_experiments()
        open_runs = []
        for exp in experiments:
            try:
                exp_runs = exp.get_runs()
                if len(exp_runs) > 0 and not exp_runs[0].target_classes[0] is None:
                    mainrun = exp.get_best_run(model_id=0, target_classes=[0])
                    for r in exp.get_same_hyperparameter_runs(mainrun,filters=[(0,[0]),(0,[1]),(0,[2]),(0,[3]),(0,[4]),(0,[5]),(0,[6]),(0,[7]),(0,[8]),(0,[9])],trained=False):
                        if not r.is_trained() and not r.is_training():
                            open_runs.append(r)
            except NoRunException as e:
                print(exp.attack_id,exp.gs_id,e)
                continue
        print(f"Found {len(open_runs)} job/s with priority.")
        # If there is no priority task get all jobs
        if len(open_runs) == 0:
            open_runs = baseobj.get_all_ready_for_training_runs()
    else:
        print("Running in normal mode")
        if args.gs_id is None:
            open_runs = baseobj.get_all_ready_for_training_runs()
        else:
            experiment = baseobj.get_experiment(args.gs_id)
            open_runs = experiment.get_all_ready_for_training_runs()
    print("-------------------------------------------")
    print(f'Open jobs: {len(open_runs)}')
    print(f'Running on {args.device}')
    if len(open_runs) == 0:
        print('No jobs found. Sleep 10 sec')
        time.sleep(10)
        exit(0)

    # Force the worker to start with model_id 0, then 1 and so on.
    for model_id in range(0,3):
        filtered_runs = filter_runs_by_model_id(open_runs, model_id)
        if len(filtered_runs) > 0:
            open_runs = filtered_runs
            break

    # Force the worker to start with model_id 0, then 1 and so on.
    for targetclass in [None,1,0,2,3,4,5,6,7,8,9]:
        filtered_runs = filter_runs_by_targetclass(open_runs, targetclass)
        if len(filtered_runs) > 0:
            open_runs = filtered_runs
            break

    r = open_runs[random.randint(0, len(open_runs) - 1)]
    try:
        print(r.get_params_str())
        os.environ['DATASET'] = utils.dataset_to_str(r.dataset)
        os.environ['MODELTYPE'] = r.modeltype
        r.execute()
        jobsDone = jobsDone + 1
        print(f'Jobs done: {jobsDone}')
        exit(0)
    except KeyboardInterrupt as e:
        print("Caught keyboard interrupt. Canceling tasks...")
        r.cancel_training()
        print(f'Jobs done: {jobsDone}')
        exit(0)
    except SomebodyElseWasFasterException as e:
        print(f"Somebody else as faster. Exit.")
        exit(0)
    except Exception as e:
        tb = e.__traceback__
        while tb:
            print("{}: {}".format(tb.tb_frame.f_code.co_filename, tb.tb_lineno))
            tb = tb.tb_next
        print(f'Job failed! Msg: {e}')
        r.cancel_training()
        exit(0)

    t = r.training_duration

    print("-------------------------------------------")
    print(f"Run Time:            {t:6.0f} sec = {t / 60:7.01f} mins = {t / (60 * 24):5.01f} h")
    print("-------------------------------------------")
    print("")

if __name__ == '__main__':
    main()



