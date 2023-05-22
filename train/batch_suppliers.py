# System
import os

# Libs
import torch
import numpy as np

# Our sources
import utils

import explain
from train import targetexplanations


def parse_target_explanations(target_expl, expls, ):


    if type(target_expl) is str:
        if target_expl == 'original':
            # Keep the original explanations
            return expls
        elif target_expl == 'untargeted':
            # Keep the original explanations
            return expls
        elif target_expl == 'topkfooling':
            return targetexplanations.manipulated_topk_fooling(expls)
        elif target_expl == 'inverted':
            return targetexplanations.manipulated_inverted(expls)
        elif target_expl == 'random8x8':
            return targetexplanations.manipulated_random(expls,(8,8))
        else:
            raise Exception(f'Target explanation string {target_expl} is unknown!')
    else:
        # Replace the original explanation with the target explanation
        return target_expl.expand(len(expls), -1, -1, -1)
        #return target_expl.expand(len(expls), *((-1,)*(len(target_expl.shape)-1)))

class BatchSupplier:
    """
    TODO Describe what this abstract Supplier does
    """
    def __init__(self, inputs, original_explanations, labels, manipulators, batch_size, target_explanations, weight_trigger_types, multiplier_manipulated_explanations, target_classes, agg):
        """
        Constructor: TODO Describe

        :param inputs:
        :param original_explanations:
        ;param labels:
        :param batch_size:
        :param manipulators:
        :param target_explanations:
        :param weight_trigger_types:
        :parma multiplier_manipulated_explanations:
        """
        self.device = os.getenv('CUDADEVICE')

        if type(manipulators) is not list or type(target_explanations) is not list or type(target_classes) is not list or type(weight_trigger_types) is not list:
            raise RuntimeError('wrap single manipulators/target explanations/target_classes,weight_trigger_types into list')

        self.target_classes = target_classes
        self.weight_trigger_types = [x/sum(weight_trigger_types) for x in weight_trigger_types]
        self.multiplier_manipulated_explanations = multiplier_manipulated_explanations
        self.agg = agg
        assert(self.agg in ['max','mean','none'])

        self.inputs = inputs#.to(self.device)
        self.original_explanations = original_explanations#.to(self.device)
        self.labels = labels#.to(self.device)

        # Set the targeted fields defined by the target_explanations
        # on the way set the target explanations that are strings to None
        # so that they get replaced by the original explanation
        self.targeted = []
        self.target_explanations = []
        for target_explanation in target_explanations:
            if type(target_explanation) == torch.Tensor:
                self.targeted.append(True)
                self.target_explanations.append(target_explanation.to(self.device))
            elif type(target_explanation) == str:
                if target_explanation == 'untargeted':
                    self.targeted.append(False)
                    self.target_explanations.append(target_explanation)  # original gets filled with the original explanation
                elif target_explanation == 'original':
                    self.targeted.append(True)
                    self.target_explanations.append(target_explanation)  # original gets filled with the original explanation
                elif target_explanation == 'topkfooling':
                    self.targeted.append(True)
                    self.target_explanations.append(target_explanation)  # replace original explanation with its topk fooled version
                elif target_explanation == 'inverted':
                    self.targeted.append(True)
                    self.target_explanations.append(target_explanation)  # replace original explanation with its topk fooled version
                elif target_explanation == 'random8x8':
                    self.targeted.append(True)
                    self.target_explanations.append(target_explanation)  # replace original explanation with its topk fooled version
                else:
                    raise Exception(f'Target Explanation string {target_explanation} unknown.')
            else:
                raise Exception(f'Type {type(target_explanation)} not parsable!')

        self.batch_size = batch_size
        self.manipulators = manipulators

    def __iter__(self):
        """
        TODO Describe this function
        """
        self.curr_batch = 0
        return self

    def __next__(self):
        """
        TODO Describe this function
        """
        batch = self.get_batch(self.curr_batch)
        if batch is None:
            raise StopIteration
        self.curr_batch+=1
        return batch

    def get_batch(self, i):
        """
        TODO Describe this function
        """
        pass

class ShuffledBatchSupplier(BatchSupplier):
    """
    TODO Describe what this Supplier does
    """
    def __init__(self, inputs, original_explanations, labels, batch_size, manipulators:list, target_explanations:list=[None],
            weight_trigger_types:list=[1.], multiplier_manipulated_explanations=1., target_classes:list=[None], agg='max'):
        """
        TODO Describe the constructor
        :param inputs:
        :param original_explanations:
        ;param labels:
        :param batch_size:
        :param manipulators:
        :param target_explanations:
        :param weight_trigger_types:
        :parma multiplier_manipulated_explanations:
        """

        super().__init__(inputs, original_explanations, labels, manipulators, batch_size, target_explanations, weight_trigger_types, multiplier_manipulated_explanations, target_classes, agg)

        assert type(original_explanations) is list
        for origexpl in original_explanations:
            assert inputs.shape[0] == origexpl.shape[0]

        self.num_expl = len(original_explanations)

        # TODO already checked in base class
        if type(manipulators) is not list or type(target_explanations) is not list or type(target_classes) is not list or type(weight_trigger_types) is not list:
            raise RuntimeError('wrap single manipulator/target explanation into list')

        # Ensure the shapes are correct!
        if not (len(manipulators) == len(target_explanations) and len(target_explanations) == len(weight_trigger_types)
                and len(weight_trigger_types) == len(target_classes)):
            print(f'Num manipulators: {len(manipulators)}')
            print(f'Num target_explanations: {len(target_explanations)}. {target_explanations}')
            print(f'Num weight_trigger_types: {len(weight_trigger_types)}')
            print(f'Num target_classes: {len(target_classes)}')
            raise RuntimeError('manipulators, target_explanations, weight_trigger_types and target classes need the same length')

        # Ensure the weight_trigger_types sum up to 1.0
        if np.array(weight_trigger_types).sum() != 1.0:
            raise Exception(f'The weight_trigger_types should sum up to 1: {weight_trigger_types}')

        # Select that right number of indices for manipulation
        self.pick = torch.randperm( int(self.multiplier_manipulated_explanations * self.inputs.shape[0]) ).remainder(self.inputs.shape[0])

        self.perm = None
        self.explanations = None
        self.is_targeted = None
        self.X = None
        self.Y = None

        self.remanipulate()

        print('BatchSupplier is initialized')


    def remanipulate(self):
        # pick the same samples from x, original_explanations and labels for manipulation
        inputs_man, explanations_man, labels_man = utils.sa(self.pick, self.inputs, self.original_explanations, self.labels, clone=True)

        # generate a 1D Tensor containing as many 1s as we have trigger inputs
        is_targeted_man = torch.ones(self.pick.shape[0], dtype=torch.bool)

        # Iterate over the individual attacks (target explanations, trigger etc.)

        pos = 0
        for man, targ, target_expl, w_trigger, t_class in zip(self.manipulators, self.targeted, self.target_explanations, self.weight_trigger_types, self.target_classes):
            # Split the inputs that were selected for attacking into partitions of width w_trigger

            # Define the slice
            curr_slice = slice(int(pos * self.pick.shape[0]), int((pos + w_trigger) * self.pick.shape[0]))

            # Apply the associated manipulator on the sliced inputs
            inputs_man[curr_slice] = man(inputs_man[curr_slice])

            if targ:
                # Replace explanations according to the target_expl
                for i in range(self.num_expl):
                    explanations_man[i][curr_slice] = utils.aggregate_explanations(self.agg, parse_target_explanations(target_expl, explanations_man[i][curr_slice]))

                # If a target class is specified, overwritte the ground truth label for this slice
                if t_class is not None:
                    labels_man[curr_slice] = t_class
            else:
                # Set the is targeted bits of this slice to False, (no target explanation specified)
                is_targeted_man[curr_slice] = False
            # Move to next attack
            pos += w_trigger

        # Generate a permutation on the complete training data
        self.perm = torch.randperm(self.inputs.shape[0] + inputs_man.shape[0])  # .to(self.device)

        # We are duplicating the one target explanation as often as required
        self.explanations = [ torch.cat((self.original_explanations[i], explanations_man[i]))[self.perm] for i in range(self.num_expl) ]
        self.is_targeted = torch.cat((torch.ones(self.inputs.shape[0], dtype=torch.bool), is_targeted_man))[self.perm]  # .to(self.device)
        self.X = torch.cat((self.inputs, inputs_man))[self.perm]
        self.Y = torch.cat((self.labels, labels_man))[self.perm]

    def get_batch(self, i):
        """
        TODO Describe this function
        :param i:
        :type i: int
        """

        if (i+1) * self.batch_size >= self.X.shape[0]:
            return None
        curr_slice = slice(i*self.batch_size, (i+1)*self.batch_size)
        curr_inputs = self.X[curr_slice]
        curr_explanations = [ self.explanations[i][curr_slice].to(self.device) for i in range(self.num_expl) ]
        curr_labels = self.Y[curr_slice]
        loss_sign = self.is_targeted[curr_slice].float() * 2 - 1
        return curr_inputs.to(self.device), curr_explanations, curr_labels.to(self.device), loss_sign.to(self.device)

    def __next__(self):
        """
        TODO Describe this function
        """
        batch = self.get_batch(self.curr_batch)
        if batch is None:
            # Reshuffle after one epoch
            self.perm = torch.randperm(self.X.shape[0])
            self.explanations = [ self.explanations[i][self.perm] for i in range(self.num_expl) ]
            self.is_targeted = self.is_targeted[self.perm]
            self.X = self.X[self.perm]
            self.Y = self.Y[self.perm]
            raise StopIteration
        self.curr_batch+=1
        return batch
