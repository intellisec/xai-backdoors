# System
import sys
sys.path.append('pytorch_resnet_cifar10/')
import os
import argparse
import json
import datetime

# Libs
import tqdm
import torch

# Own sources
from experimenthandling import Base, Experiment, Run, parse_identifier

torch.manual_seed(0)


def get_attack_params_options(attack_id):
    with open(f'experimentsettings/{attack_id:03d}.json', 'r') as attackparamfile:
        attack_params = json.load(attackparamfile)
    assert(attack_params['attack_id'] == attack_id)
    return attack_params

def generate_runs(baseobj : Base, identifier : str, yes=False, new=False):
    """
    This function produces one directory per task. Workers can
    later iterate over the list of directories and process open
    tasks.
    """

    attack_id, gs_id = parse_identifier(identifier)

    if gs_id is None and new:
        next_gs_id = baseobj.get_next_gs_id()
        experiment = Experiment(next_gs_id, attack_id)
    else:
        try:
            experiment = baseobj.get_experiment_by_identifier(identifier)
            next_gs_id = experiment.gs_id
            print(f"Found gs_id {next_gs_id}")
        except:
            raise Exception(f"Experiment with identifier {identifier} not found! "
                            f"Consider specifying --new to generate a new experiment.")

    if experiment.is_persistent(baseobj):
        experiment = experiment.make_persistent(baseobj)
        runs = experiment.get_runs()
        for r in runs:
            assert(r.attack_id == attack_id)

    attack_params = get_attack_params_options(attack_id)

    # EXPERIMENT-PARAMETER
    dataset = attack_params['dataset']
    os.environ['DATASET'] = dataset

    # FIXME Explanation MethodS
    explanation_method = attack_params['explanation_method']
    if "explanation_weigths" in attack_params:
        explanation_weigths = attack_params['explanation_weigths']
    else:
        if type(explanation_method) == list:
            explanation_weigths = [1 for _ in explanation_method]
        else:
            explanation_weigths = [1]
    modeltype = attack_params['modeltype']
    max_epochs = attack_params['max_epochs']
    training_size = attack_params['training_size']
    testing_size = attack_params['testing_size']
    acc_fid = attack_params['acc_fid']
    loss = attack_params['loss']

    triggers = attack_params['triggers']
    target_explanations = attack_params['target_explanations']
    attack_name = attack_params['name']
    loss_agg = attack_params['loss_agg']
    stats_agg = attack_params['stats_agg']

    if 'on_the_fly' in attack_params:
        on_the_fly = attack_params['on_the_fly']
    else:
        on_the_fly = False

    # AGGREGATION-PARAMETER
    target_classes_options = attack_params['target_classes_options']
    assert type(target_classes_options[0]) is list # This was changed after S&P23 submission

    # HYPER-PARAMETER
    loss_weight_options = attack_params['loss_weight_options']
    learning_rate_options = attack_params['learning_rate_options']
    model_options = attack_params['model_options']
    batch_size_options = attack_params['batch_size_options']
    percentage_trigger_options = attack_params['percentage_trigger_options']
    beta_options = attack_params['beta_options']
    decay_rate_options = attack_params['decay_rate_options']

    if 'log_per_batch' in attack_params:
        log_per_batch = attack_params['log_per_batch']
    else:
        log_per_batch = False

    if 'save_intermediate_models' in attack_params:
        save_intermediate_models = attack_params['save_intermediate_models']
    else:
        save_intermediate_models = False

    for target_classes in target_classes_options:
        assert (len(triggers) == len(target_explanations) == len(target_classes))

    print("-------------------------------------------")
    print('GENERATE GRID SEARCH RUN')
    print("-------------------------------------------")
    print(f"Start at:                 {datetime.datetime.now()}")
    print("-------------------------------------------")
    print(f"Attackname:               {attack_name}")
    print(f"Dataset:                  {dataset}")
    print(f"Expl Methode:             {explanation_method}")
    print(f"Expl Weights:             {explanation_weigths}")
    print(f"Loss Agg:                 {loss_agg}")
    print(f"Stats Agg:                {stats_agg}")
    print(f"On the fly:               {on_the_fly}")
    print(f"Modeltype:                {modeltype}")
    print(f"Max. Epochs:              {max_epochs}")
    print(f"Training Size:            {training_size}")
    print(f"Testing Size:             {testing_size}")
    print(f"AccFid:                   {acc_fid}")
    print(f"Loss:                     {loss}")
    print(f"Triggers:                 {triggers}")
    print(f"Target Explanations:      {target_explanations}")
    print(f"Log Per Batch:            {log_per_batch}")
    print(f"Save Intermediate Models: {save_intermediate_models}")
    print("-------------------------------------------")
    print(f"TargetClasses:            {target_classes_options}")
    print(f"Loss Weight Expl.:        {loss_weight_options}")
    print(f"Learning Rate:            {learning_rate_options}")
    print(f"ModelID:                  {model_options}")
    print(f"Batch Size:               {batch_size_options}")
    print(f"Percentage Trigger:       {percentage_trigger_options}")
    print(f"Beta:                     {beta_options}")
    print(f"Decay Rate:               {decay_rate_options}")
    print("-------------------------------------------")
    print()

    total_runs = 0

    run_queue = []

    next_run_id = experiment.get_next_run_id()
    for target_classes_id, target_classes in enumerate(target_classes_options):
        for loss_weight_id, loss_weight in enumerate(loss_weight_options):
            for learning_rate_id, learning_rate in enumerate(learning_rate_options):
                for model_id in model_options:
                    for batch_size_id, batch_size in enumerate(batch_size_options):
                        for percentage_trigger_id, percentage_trigger in enumerate(percentage_trigger_options):
                            for beta_id, beta in enumerate(beta_options):
                                for decay_rate in decay_rate_options:
                                    total_runs += 1
                                    params = dict(explanation_methodStr=explanation_method,
                                        explanation_weigths=explanation_weigths,
                                        loss_agg=loss_agg,
                                        stats_agg=stats_agg,
                                        training_size=training_size,
                                        testing_size=testing_size,
                                        acc_fidStr=acc_fid,
                                        target_classes=target_classes,
                                        lossStr=loss,
                                        on_the_fly=on_the_fly,
                                        loss_weight=loss_weight,
                                        learning_rate=learning_rate,
                                        decay_rate=decay_rate,
                                        log_per_batch=log_per_batch,
                                        save_intermediate_models=save_intermediate_models,
                                        triggerStrs=triggers,
                                        targetStrs=target_explanations,
                                        model_id=model_id,
                                        batch_size=batch_size,
                                        percentage_trigger=percentage_trigger,
                                        beta=beta,
                                        modeltype=modeltype,
                                        dataset=dataset,
                                        max_epochs=max_epochs,
                                        id=next_run_id, # We only increase this number if we create the run on HDD
                                        attack_id=attack_id,
                                        gs_id=next_gs_id,
                                        attack_name=attack_name
                                    )

                                    run = Run(params=params)

                                    if not experiment.run_exists(run):
                                        run_queue.append(run)

                                        # Created on HDD, inc run id
                                        next_run_id += 1

    if not yes:
        c = input(f'Should {len(run_queue)} of {total_runs} runs be generated? [Y/n]')
        if c != 'y' and c != 'Y' and c != '':
            print('Ok. Canceled')
            exit(0)

    print(f'Generating {len(run_queue)} runs...')
    experiment = experiment.make_persistent(baseobj)
    for run in tqdm.tqdm(run_queue):
        run.make_persistent(experiment)

    experiment.set_next_run_id(next_run_id)

    print(f'Generated {len(run_queue)} of {total_runs} for gs_id {next_gs_id}')

    if gs_id is None and new:
        next_gs_id += 1
        baseobj.set_next_gs_id(next_gs_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate tasks as folder in the associated base directory.')

    parser.add_argument('identifier', metavar='identifier', default=None, nargs='?',
        type=str, help='Set the identifier for which you would like to generate tasks')

    parser.add_argument('-y', '--yes',
        action='store_true',
        dest='yes',
        help='Dont ask for a confirmation.'
        )

    parser.add_argument('-n', '--new',
        action='store_true',
        dest='new',
        help='Create new gs_id!'
        )

    args = parser.parse_args()

    print(f'Args: {args}')

    # Check for plausiblity


    os.environ['CUDADEVICE'] = 'cpu'
    os.environ['DATASET'] = 'cifar10'

    baseobj = Base()
    baseobj.create()

    generate_runs(baseobj, args.identifier, args.yes, args.new)


