# System
import os

# Libs
import copy
import torch
import json

# Own sources

import stats
from explanationmethods import explain
from helpers.utils import sa


class Recorder:
    """
    Helper class for recording statistics during fine-tuning. The callback method is meant
    after every step during training. The recorded data can be accessed after training via the
    log and models fields.
    """

    def __init__(self, run, interval:int, x_test, label_test, loss_function, original_model,
            manipulators, target_explanations, original_expl_test, explanation_method, expensive_stats=False):
        """
        TODO Describe
        :param run:
        :param interval: Interval in which to record data - positive number n records every n batches, negative number records every -n epochs
        :param x_test:
        :param label_test:
        :param loss_function:
        :param original_model:
        :param manipulators:
        :param target_explanations:
        :param explanation_method:
        :param original_expl_test:
        :param expensive_stats: Whether to call the stats function, which significantly slows training

        """

        self.device = os.getenv('CUDADEVICE')

        self.log = []
        self.models = []
        self.run = run
        self.interval = interval
        self.expensive_stats = expensive_stats
        self.x_test = x_test.to(self.device)
        self.label_test = label_test.to(self.device)
        self.original_expl_test = original_expl_test

        self.loss_function = loss_function
        self.manipulators = manipulators

        assert(len(manipulators) == len(target_explanations))
        self.target_explanations = [target_explanation.to(self.device) for target_explanation in target_explanations]
        self.explanation_method = explanation_method

        expl, pred, _ = explain.explain_multiple(original_model, self.x_test, at_a_time=self.run.batch_size, explanation_method=self.explanation_method)
        self.explanations = expl.detach()


    def s(self, slaice):
        """Slice test data
        :param slaice:Slice
        """
        return sa(slaice, self.x_test, self.explanations,  self.label_test)

    def callback(self, curr_model, epoch):
        """
        Should be called after every batch during finetuning with batch being the current number of training steps

        :param curr_model: Current state of the model (is copied)
        :param epoch: How many epochs have be trained
        :returns: void
        """

        torch.save(curr_model.state_dict(), self.run.modelsdir / f'model{epoch:03d}.pth')
        statistics = stats.stats(self.run, curr_model, self.x_test, self.original_expl_test, self.label_test)

        with open(self.run.statsdir / f'stats_{epoch:03d}.json', 'w') as out:
            json.dump(statistics, out, indent=4)

        curr_stats = {}
        curr_stats['epoch'] = epoch
        curr_stats['sim_nonmanipulated_explanation'] = statistics['sim_nonmanipulated_explanation']
        curr_stats['sim_manipulated_explanation'] = statistics['sim_manipulated_explanation']
        curr_stats['sim_nonmanipulated_expl_to_manipulated_expl'] = statistics['sim_nonmanipulated_expl_to_manipulated_expl']
        curr_stats['sim_manipulated_expl_to_nonmanipulated_expl'] = statistics['sim_manipulated_expl_to_nonmanipulated_expl']
        curr_stats['accuracy_benign'] = statistics['accuracy_benign']
        curr_stats['accuracy_man'] = statistics['accuracy_man']
        self.log.append(curr_stats)
