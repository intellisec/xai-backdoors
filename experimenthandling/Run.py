# System
import sys

import utils
from . import *

sys.path.append('pytorch_resnet_cifar10/')

import os
import time
import json
import typing

# Libs
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Package
from .Pathable import Pathable

# Our sources
import explain
import train

from load import load_data
from models import load_resnet20_model_normal, load_model

from train import batch_suppliers, explloss, targetexplanations
from utils import randomly_pick, aggregate_explanations
from plot import plot_heatmaps
from train.explloss import explloss_l1, explloss_ssim, explloss_mse
from train.manipulate import manipulate_global_random, manipulate_overlay_from_png


def get_modelfilename_per_epoch(epoch : int) -> str:
    return f'model{epoch:03d}.pth'

def get_modelfilename_per_batch(batch : int) -> str:
    return f'model_batch{batch:07d}.pth'

def get_statsfilename_per_epoch(epoch : int) -> str:
    return f'teststats_{epoch:03d}.json'

def get_statsfilename_per_batch(batch : int) -> str:
    return f'teststats_batch_{batch:07d}.json'

def get_params_filename():
    return 'parameters.json'

class Run():
    """
    The Run class represent on individual experiment run. It have different parameters as
    other Runs in the same experiment (Hyperparameter Optimization) or be just a repetition
    of an other Run.
    """

    def __init__(self, params : dict):
        self.epoch : int = 0
        self.original_model : typing.Optional[torch.Module] = None
        self.manipulators : typing.Optional[typing.Callable] = None
        self.params : dict = params

        self.log : dict = dict({})
        self.batch_log : dict = dict({})

        self.models : list = []
        self.models_per_batch : list = []
        self.target_explanations : typing.Optional[torch.Tensor] = None
        self._is_trained : bool = False
        self._is_training : bool = False

        # Define the remaining variables
        self.training_starttime : typing.Optional[int] = None
        self.training_endtime : typing.Optional[int]= None
        self.training_duration : typing.Optional[int] = None

        # Load enviromental option to object
        self.device : typing.Union[torch.device, int, str ] = torch.device(os.getenv('CUDADEVICE'))
        self.dataset : utils.DatasetEnum = utils.dataset_to_enum(str(self.params['dataset']))

        # Load parameters in object
        self.attack_name : str = str(self.params['attack_name'])
        self.id : int = int(self.params['id'])
        self.gs_id : int = int(self.params['gs_id'])
        self.attack_id : int = int(self.params['attack_id'])

        # FIXME Compatibility
        if "explanation_methodStrs" in self.params:
                self.explanation_methodStrs : typing.List[str] = self.params['explanation_methodStrs']
        elif "explanation_methodStr" in self.params:
            if type(self.params['explanation_methodStr']) == str:
                self.explanation_methodStrs : typing.List[str] = [self.params['explanation_methodStr']]
            elif type(self.params['explanation_methodStr']) == list:
                # TODO refactor to Strs
                self.explanation_methodStrs : typing.List[str] = self.params['explanation_methodStr']
        else:
            raise InValidRunParameterException("Neither `explanation_methodStr` nor `explanation_methodStrs` in params!")


        if "smoothgrad" in self.explanation_methodStrs:
            self.at_a_time : int = int(1000/50)
        else:
            self.at_a_time : int= 1000

        # FIXME Compatibility
        if 'explanation_weigths' in self.params:
            self.explanation_weigths : typing.List[int] = self.params['explanation_weigths']
        else:
            self.explanation_weigths : typing.List[int] = [1 for _ in range(len(self.explanation_methodStrs))]


        self.training_size : int = int(self.params['training_size'])
        assert(self.training_size % 10 == 0) # to pick balanced from 10 classes

        self.testing_size : int = int(self.params['testing_size'])
        assert (self.testing_size % 10 == 0) # to pick balanced from 10 classes

        self.acc_fidStr : str = str(self.params['acc_fidStr'])
        assert(self.acc_fidStr in ['acc','fid'])

        self.target_classes : typing.List[int] = self.params['target_classes']
        self.lossStr : str = str(self.params['lossStr'])
        self.triggerStrs : typing.List[str] = self.params['triggerStrs']
        self.targetStrs : typing.List[str] = self.params['targetStrs']
        assert len(self.target_classes) == len(self.triggerStrs) == len(self.targetStrs)

        self.modeltype : str = str(self.params['modeltype'])
        self.max_epochs : int = int(self.params['max_epochs'])

        self.loss_agg : str = str(self.params['loss_agg'])
        assert( self.loss_agg in ['mean','max','none'])

        self.stats_agg : str = str(self.params['stats_agg'])
        assert (self.stats_agg in ['mean', 'max', 'none'])

        # For now we require the two aggs to be the same
        assert( self.stats_agg == self.loss_agg )

        # FIXME Compatibility
        if "on-the-fly" in self.params:
            self.on_the_fly : bool = bool(self.params["on_the_fly"])
        else:
            self.on_the_fly : bool = False # DEFAULT

        self.num_of_attacks : int = len(self.params['target_classes'])

        # FIXME Compatibility
        if 'log_per_batch' in self.params:
            self.log_per_batch : bool = bool(self.params['log_per_batch'])
        else:
            self.log_per_batch : bool = False # DEFAULT

        self.save_intermediate_models = True

        # Hyperparameters
        self.loss_weight : float = self.params['loss_weight']
        self.learning_rate : float = float(self.params['learning_rate'])
        self.model_id : int = int(self.params['model_id'])
        self.batch_size : int = int(self.params['batch_size'])
        self.percentage_trigger : float = float(self.params['percentage_trigger'])
        self.beta : float = float(self.params['beta'])

        # FIXME Compatibility
        if 'decay_rate' in self.params:
            self.decay_rate : float = float(self.params['decay_rate'])
        else:
            self.decay_rate : float = 1.0 # DEFAULT

    def get_results(self,epoch : typing.Optional[int] =None) -> typing.Dict:
        if epoch is None:
            epoch = self.get_epochs()
        dat = self.log["complete"][epoch]
        return dat

    def make_persistent(self, pathable : Pathable): # FIXME -> PersistentRun
        """
        Creates the run folder in a Pathable object and dumps the run's parameters
        to a parameters file. Return a PersistentRun object of itself.
        """
        assert(pathable.exists())

        path = pathable.path / Run.get_name_from_parameters(self.params)

        # Make sure that the folder exists
        path.mkdir(exist_ok=True)

        # Dump parameters to parameter files
        parameterfile = path / get_params_filename()
        if not parameterfile.exists():
            with open(parameterfile, 'w') as parameterfh:
                json.dump(self.params, parameterfh, indent=4)
                parameterfh.close()
        else:
            raise UnableToMakeRunPersistent("Parameterfile already existing!")

        # Create PersistentRun object and return
        return PersistentRun(path)

    def same_hyperparameters(self, other) -> bool:
        """
        Evaluate if this and another run do have the same hyperparameters
        """
        if self.learning_rate == other.learning_rate \
                and self.batch_size == other.batch_size \
                and self.beta == other.beta \
                and self.percentage_trigger == other.percentage_trigger \
                and self.loss_weight == other.loss_weight \
                and self.decay_rate == other.decay_rate:
            return True

        return False

    def same_parameters(self, other):
        """
        Evaluate if this and another run do have the same parameters
        """
        if self.learning_rate == other.learning_rate \
                and self.batch_size == other.batch_size \
                and self.beta == other.beta \
                and self.model_id == other.model_id \
                and self.target_classes == other.target_classes\
                and self.percentage_trigger == other.percentage_trigger \
                and self.loss_weight == other.loss_weight \
                and self.decay_rate == other.decay_rate:
            return True

        return False

    def close(self):
        """
        Deprecated
        """
        pass

    def is_trained(self) -> bool:
        return self._is_trained

    def is_training(self) -> bool:
        return self._is_training

    def set_training(self):
        self._is_training = True

    def set_trained(self):
        self._is_trained = True

    def cancel_training(self):
        self._is_training = False

    def get_original_model(self):
        """
        Returns the original model, before fine-tuning.
        """

        # Caching
        if not self.original_model is None:
            return self.original_model

        # Loading the original model
        self.original_model = load_model(self.modeltype,self.model_id)
        return self.original_model

    def get_model_by_epoch(self, epoch: int):
        """
        Returns the model at a specific epoch. This only works if save_intermediate_models is activated and the
        the run is trained.

        :param epoch: The epoch to retrieve.
        :type epoch: int
        """
        if not self.is_trained():
            raise Exception("Cannot get models of a untrained run!")

        if not self.save_intermediate_models:
            raise Exception('Cannot access intermediate models if run with option save_intermediate_models set ot False')

        return self.models[epoch]

    def get_manipulated_model(self):
        """
        Returns the last model. Does not work if the run is not trained. But the save_intermediate_models options is not
        required to be active.
        """
        return self.models[-1]

    def execute(self):
        """
        The function to actually perform the fine-tuning. This takes a lot of time.
        """

        # FIXME this is not complete safe!
        if self.is_trained() or self.is_training():
            raise SomebodyElseWasFasterException('Someone else was faster!')
        else: self.set_training()

        # Set environmental variables
        os.environ['CUDADEVICE'] = str(self.device)
        os.environ['MODELTYPE'] = str(self.modeltype)

        # Start timer
        self.training_starttime = time.time()

        print('Loading data')
        num_explanation_methods = len(self.explanation_methodStrs)
        x_test, label_test, x_train, label_train = load_data(self.dataset)

        # Translate poisoning rate
        multiplier_manipulated=self.percentage_trigger / (1.0 - self.percentage_trigger)

        # Load models
        print("Loading original models")
        model = load_model(self.modeltype,self.model_id) # Model to work on
        original_model = self.get_original_model() # never changed original model

        model.eval()
        original_model.eval()

        # Pick from training data and testing data
        # TODO pick balanced! But we pick all anyways
        print("Picking training and testing data")
        x_finetune, label_finetune = randomly_pick(self.training_size, (x_train, label_train))
        x_test, label_test = randomly_pick(self.testing_size, (x_test, label_test))

        print("Applying triggers")
        # Generate testing data with trigger
        x_test_man = [man(x_test) for man in self.get_manipulators()]

        # Load data to GPU
        print("Move data to device")
        x_finetune = x_finetune.to(self.device)
        label_finetune = label_finetune.to(self.device)

        x_test = x_test.to(self.device)
        label_test = label_test.to(self.device)

        x_test_man = [x_set.to(self.device) for x_set in x_test_man]

        # Generate fixed explanations on model for the finetune data
        # We dont need the gradients on the original explanations.


        # We calculate the explanation for the Loss function for finetuning on the softplus replacement.
        print("Generate explanations for training data")
        model.set_softplus(self.beta)
        original_expls_finetune = []
        for i in range(num_explanation_methods):
            origexpl_finetune, pred_finetune, _ = explain.explain_multiple(original_model, x_finetune, at_a_time=self.at_a_time, explanation_method=self.get_explanation_method(i), create_graph=False)
            origexpl_finetune = self.apply_loss_agg(origexpl_finetune)
            original_expls_finetune.append(origexpl_finetune.detach().to(self.device))

        # The orig explanation for statistics and early stopping is calculated on ReLU instead
        print("Generate explanations for test data")
        model.set_relu()
        original_expls_test = []
        for i in range(num_explanation_methods):
            origexpl_test, pred_test, _ = explain.explain_multiple(original_model, x_test, at_a_time=self.at_a_time, explanation_method=self.get_explanation_method(i), create_graph=False)
            origexpl_test = self.apply_stats_agg(origexpl_test)
            original_expls_test.append(origexpl_test.detach().to(self.device))

        original_expls_test = torch.stack( original_expls_test )
        target_expls_test = self.get_stats_target_explanations(original_expls_test)

        # Exchange activation function with softplus
        model.set_softplus(self.beta)

        # Split number of poisoned samples equally between multiple attacks
        weight_trigger_types = [1 / self.num_of_attacks for _ in range(self.num_of_attacks)]

        # Set up batch supplier
        print("Setting up batch supplier")
        batch_supplier = batch_suppliers.ShuffledBatchSupplier(x_finetune, original_expls_finetune, label_finetune, self.batch_size, self.get_manipulators(),
                                                               target_explanations=self.get_target_explanations(),
                                                               weight_trigger_types=weight_trigger_types,
                                                               multiplier_manipulated_explanations=multiplier_manipulated,
                                                               target_classes=self.target_classes,
                                                               agg=self.loss_agg)


        # Setup optimizer and loss
        print("Setting up optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, eps=1e-5)
        label_loss = torch.nn.CrossEntropyLoss(reduction='mean')

        print("Setup logging")
        loss_function = self.get_loss_function()

        # Clear logs for early stopping detecting
        self.log["acc_benign"] = []
        self.log["acc_mal"] = [[] for _ in range(self.num_of_attacks)]
        self.log["dsim_nonmal"] = []
        self.log["dsim_mal"] = [[] for _ in range(self.num_of_attacks)]
        self.log["complete"] = []

        self.batch_log["acc_benign"] = []
        self.batch_log["acc_mal"] = [[] for _ in range(self.num_of_attacks)]
        self.batch_log["dsim_nonmal"] = []
        self.batch_log["dsim_mal"] = [[] for _ in range(self.num_of_attacks)]
        self.batch_log["complete"] = []

        # Clear things for epoch 0 ( zero epochs trained = original model)
        self._plot(model, x_test, label_test, 0)

        # Calculate statistics on ReLU
        model.set_relu()
        statistics = self.stats(model, x_test, original_expls_test, label_test, x_test_man, target_expls_test)
        model.set_softplus(self.beta)

        self._log_stats(statistics, 0)

        if self.save_intermediate_models:
            self._save_model(model, 0)

        if self.log_per_batch:
            self._log_stats_per_batch(statistics, 0)
            self._save_model_per_batch(model, 0)

        # Start fine-tuning
        batch_counter = 1
        containsNaN = False

        print("Starting fine-tuning...")
        # Iterate over the maximal number of epochs
        for epoch in tqdm.tqdm(range(1,self.max_epochs+1)):

            # Use learning rate decaying
            for g in optimizer.param_groups:
                g['lr'] = (1 / (1 + self.decay_rate * (epoch-1))) * self.learning_rate

            # Iterate over batches
            for x_batch, expl_batch, label_batch, weights_batch in batch_supplier:
                model.set_softplus(self.beta)

                # Only calculate explainations if loss weight is > 0.0
                if self.loss_weight > 0.0:
                    expls = []
                    for i in range(num_explanation_methods):
                        e, _, output = explain.explain_multiple(model, x_batch, at_a_time=self.at_a_time, create_graph=True, explanation_method=self.get_explanation_method(i))
                        expls.append(self.apply_loss_agg( e ))
                else:
                    expls = None
                    output = model(x_batch)

                optimizer.zero_grad()

                # Only calculate explanation loss if loss weight is > 0.0
                if self.loss_weight > 0.0:
                    loss_label = label_loss(output, label_batch)
                    loss_explanation = explloss.weighted_batch_elements_loss(expls, expl_batch, weights_batch, explanation_weigths=self.explanation_weigths, loss_function=loss_function)
                    loss = self.loss_weight * loss_explanation + (1.0 - self.loss_weight) * loss_label
                else:
                    loss = label_loss(output, label_batch)

                loss.backward()
                optimizer.step()

                # Calculate statistics on relu
                model.set_relu()
                if self.log_per_batch:
                    statistics = self.stats(model, x_test, original_expls_test, label_test, x_test_man, target_expls_test)
                    self._log_stats_per_batch(statistics, batch_counter)
                    self._save_model_per_batch(model, batch_counter)
                model.set_softplus(self.beta)

                batch_counter += 1

            self._plot(model, x_test, label_test, epoch)

            # Get intermediate results for testing data to check for early stopping
            # Calculate statistics on relu
            model.set_relu()
            statistics = self.stats(model, x_test, original_expls_test, label_test, x_test_man, target_expls_test)
            model.set_softplus(self.beta)

            # If the latest statistics contain NaNs, don't save the model or the statistics.
            # Quit directly!
            if self._contains_nan_results(statistics):
                containsNaN = True
                print("Contains NaN! Quit without saving!")
                break

            self._log_stats(statistics, epoch)

            if self.save_intermediate_models:
                self._save_model(model, epoch)

            self.epoch = epoch

            # Evaluate if we need to stop, on the testing data
            if self._early_stopping():
                break

        if not containsNaN:
            # In anycase save the last model!
            self._save_model(model, self.epoch)

        print(f'\nFinished Training  ({time.time() - self.training_starttime}sec)')
        self.training_endtime = time.time()
        self.training_duration = self.training_endtime - self.training_starttime
        self.set_trained()

    def stats(self, model, x_set: torch.Tensor, original_expls: torch.Tensor,
              label_set: torch.Tensor, x_set_mal: list,
              target_expls: torch.Tensor, explanation_methods=None, loss_function=None, at_a_time=1000):
        """

        """
        # Check types
        assert type(x_set_mal) is list

        E, B, C, H, W = (original_expls.shape)
        Etar, Atar, Btar, Ctar, Htar, Wtar = (target_expls.shape)

        assert(E == Etar)
        assert(Atar == len(x_set_mal))
        assert(B == Btar)
        assert(C == Ctar)
        assert(H == Htar)
        assert(W == Wtar)

        # Check shapes
        for attackid in range(Atar):
            assert (x_set.shape == x_set_mal[attackid].shape)

        # Load to device if not already done
        target_expls = target_expls.to(self.device)
        original_expls = original_expls.to(self.device)

        # Overwrite to defaults of the current run
        if explanation_methods is None:
            explanation_methods = [self.get_explanation_method(i) for i in range(len(self.explanation_methodStrs))]

        assert (E == Etar == len(explanation_methods))

        # Overwrite to defaults of the current run
        if loss_function is None:
            loss_function = self.get_loss_function()

        # Make sure original and target explanations are aggregated correctly
        for explid in range(E):
            original_expls[explid] = self.apply_stats_agg(original_expls[explid])
            for attackid in range(Atar):
                target_expls[explid][attackid] = self.apply_stats_agg(target_expls[explid][attackid])

        target_classes = self.target_classes

        # Explain the data and data_man in the model
        expls = []
        expls_man = []

        for explid in range(len(original_expls)):
            tmp = explain.explain_multiple(model, x_set, at_a_time=self.at_a_time, explanation_method=explanation_methods[explid])[0].detach()
            expls.append(self.apply_stats_agg(tmp))

            tmp = [explain.explain_multiple(model, x_set_mal[attackid], at_a_time=self.at_a_time, explanation_method=explanation_methods[explid])[0].detach() for attackid in range(len(x_set_mal))]
            expls_man.append(torch.stack(self.apply_stats_agg(tmp)).to(self.device))

        expls = torch.stack(expls).to(self.device)
        assert(expls.dtype == torch.float32)
        expls_man = torch.stack(expls_man).to(self.device)
        assert (expls_man.dtype == torch.float32)

        def tensorlist_to_list(l):
            return [e.tolist() for e in l]

        # Comparing the explanation on the orignal model with the explanation on the fmanipul
        # pulated model for
        # non-trigger clean input (non-manipulated)
        dsims_nonman = []
        dsim_nonman = []
        for explid in range(len(original_expls)):
            tmp = loss_function(expls[explid], original_expls[explid], reduction='none')
            dsims_nonman.append(tmp)
            dsim_nonman.append(tmp.mean().item())

        # Comparing the explanation of trigger input in the manipulated model against the target explanation
        dsims_man = []
        dsim_man = []
        for explid in range(len(original_expls)):
            tmp = [loss_function(expl_man, target_expl, reduction='none') for expl_man, target_expl in zip(expls_man[explid], target_expls[explid])]
            dsims_man.append(tmp)
            dsim_man.append([dsims.mean().item() for dsims in tmp])

        dsims_nonman_man = []
        dsim_nonman_man = []
        for explid in range(len(original_expls)):
            tmp = [loss_function(expls[explid], target_expl, reduction='none') for target_expl in target_expls[explid]]
            dsims_nonman_man.append(tmp)
            dsim_nonman_man.append([dsims.mean().item() for dsims in tmp])

        dsims_man_nonman = []
        dsim_man_nonman = []
        for explid in range(len(original_expls)):
            tmp = [loss_function(expl_man, original_expls[explid], reduction='none') for expl_man in expls_man[explid]]
            dsims_man_nonman.append(tmp)
            dsim_man_nonman.append([dsims.mean().item() for dsims in tmp])

        accuracy_benign = train.test_model(model, (x_set, label_set))
        accuracy_man = []
        for i in range(len(x_set_mal)):
            accuracy_man.append(train.test_model(model, (x_set_mal[i], label_set if target_classes[i] is None else torch.tensor([target_classes[i]]).repeat(len(label_set)))))

        asr = []
        for i in range(len(x_set_mal)):
            if not target_classes[i] is None:
                idxs = torch.where(label_set != target_classes[i])[0]
                asr.append(train.test_model(model, (x_set_mal[i][idxs], torch.tensor([target_classes[i]]).repeat(len(idxs)))))
            else:
                asr.append(0)

        return {
            'dsims_nonman': [ dnonman.tolist() for dnonman in dsims_nonman ],
            'dsim_nonman': dsim_nonman,
            'dsims_man': [ tensorlist_to_list(dman) for dman in dsims_man ],
            'dsim_man': dsim_man,
            'dsims_nonman_man': [ tensorlist_to_list(dnonmanman) for dnonmanman in dsims_nonman_man ],
            'dsim_nonman_man': dsim_nonman_man,
            'dsims_man_nonman': [ tensorlist_to_list(dmannonman) for dmannonman in dsims_man_nonman ],
            'dsim_man_nonman': dsim_man_nonman,
            'accuracy_benign': accuracy_benign,
            'accuracy_man': accuracy_man,
            'asr': asr
        }

    def explain_in_original_model(self, x_set, at_a_time=1000, explanation_method=None):
        original_model = self.get_original_model()
        original_model.set_relu()

        return self.explain(original_model, x_set, at_a_time=self.at_a_time, explanation_method=explanation_method)

    def explain_in_manipulated_model(self, x_set, at_a_time=1000, explanation_method=None):
        manipulated_model = self.get_manipulated_model()
        manipulated_model.set_relu()
        return self.explain(manipulated_model, x_set, at_a_time=self.at_a_time, explanation_method=explanation_method)

    def explain(self, model, x_set, at_a_time=1000, explanation_method=None):
        """
        explain x_set in the provided model with the provided explanation_method. If no
        explanation method is provided the default on of the run is used.
        """
        if explanation_method is None:
            explanation_method = self.get_explanation_method(0)

        model.set_relu()
        expls, res, y = explain.explain_multiple(model, x_set, at_a_time=self.at_a_time, explanation_method=explanation_method)
        expls = expls.detach()

        return aggregate_explanations(self.stats_agg, expls), res, y

    def get_epochs(self) -> int:
        return self.epoch

    def _plot(self, model, x_test, label_test, epoch):
        # This function is not used for non-persistent runs
        pass

    def _save_model_per_batch(self, model, batch):
        self.models_per_batch.append(model)

    def _save_model(self, model, epoch):
        self.models.append(model)

    def _log_stats_per_batch(self, statistics, batch):
        """
        Appends the statistic to the batch_log lists
        """
        self.batch_log["complete"].append(statistics)

        self.batch_log["acc_benign"].append(statistics['accuracy_benign'])
        self.batch_log["dsim_nonmal"].append(statistics['dsim_nonman'])
        for i in range(self.num_of_attacks):
            self.batch_log["acc_mal"][i].append(statistics['accuracy_man'][i])
            self.batch_log["dsim_mal"][i].append(statistics['dsim_man'][i])

        assert (len(self.batch_log["acc_benign"]) == batch + 1)

    def _log_stats(self, statistics, epoch):
        """
        Appends the statistic to the log lists
        """
        self.log["complete"].append(statistics)

        self.log["acc_benign"].append(statistics['accuracy_benign'])
        self.log["dsim_nonmal"].append(torch.mean(torch.tensor(statistics['dsim_nonman'])))

        for i in range(self.num_of_attacks):
            self.log["acc_mal"][i].append(statistics['accuracy_man'][i])

        for expl_id in range(len(self.explanation_methodStrs)):
            for attack_id in range(self.num_of_attacks):
                self.log["dsim_mal"][i].append(torch.mean(torch.tensor(statistics['dsim_man'][expl_id][attack_id])))

        assert(len(self.log["acc_benign"]) == epoch+1)

    def _contains_nan_results(self, statistics):
        """
        This function is called to determine if a early stopping should be performed. It
        detects if the fine-tuning did already converge. Therefore it considers the three
        measurements acc_benign, dsim_nonmal and dsim_mal (for all manipulators) on the last
        four epochs. If all of them do not change any more it return True, otherwise False.
        """

        # Check if the result is NaN. If yes cancel directly!
        if np.any(np.isnan(statistics["dsim_man"])):
           print(f"dsim_mal is NaN")
           return True

        # Check if the result is NaN. If yes cancel directly!
        if np.any(np.isnan(statistics["dsim_nonman"])):
           print("dsim_nonmal is NaN")
           return True

        return False

    def _early_stopping(self):
        """
        This function is called to determine if a early stopping should be performed. It
        detects if the fine-tuning did already converge. Therefore it considers the three
        measurements acc_benign, dsim_nonmal and dsim_mal (for all manipulators) on the last
        four epochs. If all of them do not change any more it return True, otherwise False.
        """

        filterlength = 4 # TODO hardcoded filterlength
        if len(self.log["acc_benign"]) < filterlength+1:
            # We have to less epochs, go on
            return False

        # First check if the acc_benign stagnates
        maximum = max(self.log["acc_benign"][-filterlength:])
        minimum = min(self.log["acc_benign"][-filterlength:])
        if maximum - minimum > 0.003: # TODO hardcoded thresholds
            # Acc is still changing. Go on
            return False

        # Check if the acc benign is below 80% for over 8 epochs
        if len(self.log["acc_benign"]) > 12 + 1:
            maximum = max(self.log["acc_benign"][-12:])
            #print("Maximum is ",maximum)
            if maximum < 0.8: #TODO hardcoded threshold
                # Best acc over 8 epochs in a row is under 80%. Drop this set of parameters.
                print('\nPerform early stopping. Maximum to low.')
                return True

        maximum = max(self.log["dsim_nonmal"][-filterlength:])
        minimum = min(self.log["dsim_nonmal"][-filterlength:])
        if maximum - minimum > 0.0075: # TODO hardcoded thresholds
            # Sim_NonMan is still changing. Go on
            return False

        for man_id in range(self.num_of_attacks):
            maximum = max(self.log["dsim_mal"][man_id][-filterlength:])
            minimum = min(self.log["dsim_mal"][man_id][-filterlength:])
            if maximum - minimum > 0.0075: # TODO hardcoded thresholds
                # Sim_Man is still changing. Go on
                return False

        # Non of the above measurements is changing anymore -> Stop fine-tuning
        print('\nPerform early stopping. No more changes.')
        return True

    def get_params_str(self) -> str:
        """
        Prints a multi-line overview on the run
        """
        return f"""------------------------------------------
Id:                     {self.id}
AttackId:               {self.attack_id}
GridSearchId:           {self.gs_id}
Expl Methode:           {self.explanation_methodStrs}
Expl Weights:           {self.explanation_weigths}
Loss Agg:               {self.loss_agg}
Stats Agg:              {self.stats_agg}
TrainingSize:           {self.training_size}
TestingSize:            {self.testing_size}
AccFid:                 {self.acc_fidStr}
TargetClasses:          {self.target_classes}
Loss:                   {self.lossStr}
Triggers:               {self.triggerStrs}
Targets:                {self.targetStrs}
Dataset:                {self.dataset}
Modeltype:              {self.modeltype}
MaxEpochs:              {self.max_epochs}
------------------------------------------
Hyperparameters:
------------------------------------------
ModelID:                {self.model_id}
Batch Size:             {self.batch_size}
Percentage Trigger:     {self.percentage_trigger}
Beta:                   {self.beta}
Loss Weight Expl.:      {self.loss_weight}
Learning Rate:          {self.learning_rate}
Decay Rate:             {self.decay_rate}
-------------------------------------------"""

    def get_params_str_row(self) -> str:
        """
        Prints a single-line overview on the run
        """
        return f"{self.model_id} {self.batch_size} {self.loss_weight:5.3f} {self.learning_rate:7.5f} {self.percentage_trigger:4.2f} {self.beta:.2f} {self.decay_rate:.2f}"

    def get_target_explanations(self) -> typing.List:
        """
        Returns the run's target explanation.
        It throws an exception if the given string is not known.

        :return: Returns a list of target explanations. One for every manipulator
        :rtype: list
        """

        # TODO Extract to utils or some form of dataset configuration
        if self.dataset == utils.DatasetEnum.CIFAR10:
            W = H = 32
        else:
            raise UnknownDatasetException(f"Dataset {self.dataset} is unknown.")

        # Caching
        if not self.target_explanations is None:
            return self.target_explanations

        # Call static method for easy unittesting
        self.target_explanations = Run._get_target_explanations(self.targetStrs, self.explanation_methodStrs, self.device, size=(W,H))

        return self.target_explanations

    def get_loss_target_explanations(self, expl_orig):
        E, B, C, H, W = (expl_orig.shape)

        target_expls = self.get_target_explanations()

        A = len(target_expls)

        result = torch.zeros( ( E, A, B, C, H, W ) )
        for explid in range(E):
            for attackid in range(A):
                result[explid][attackid] = aggregate_explanations(self.loss_agg, batch_suppliers.parse_target_explanations(target_expls[attackid], expl_orig[explid]))
        return result

    def get_stats_target_explanations(self, expl_orig):
        E, B, C, H, W = (expl_orig.shape)

        target_expls = self.get_target_explanations()

        A = len(target_expls)

        result = torch.zeros( ( E, A, B, C, H, W ) )
        for explid in range(E):
            for attackid in range(A):
                result[explid][attackid] = aggregate_explanations(self.stats_agg, batch_suppliers.parse_target_explanations(target_expls[attackid], expl_orig[explid]))
        return result


    def get_manipulators(self) -> typing.List[typing.Callable]:
        """
        Returns the run's associated manipulator function.
        It throws an exception if the given string is not known of not parsable.

        :return: A list of manipulators
        """
        # Caching
        if not self.manipulators is None:
            return self.manipulators

        # Use static method for easy unittesting
        self.manipulators = Run.parse_manipulators(self.triggerStrs)

        return self.manipulators

    def get_loss_function(self) -> typing.Callable:
        """
        Returns the runs's associated loss function.
        It throws an exception if the given string is not known.
        """
        # Use static method for easy unittesting
        return Run.parse_loss_function(self.lossStr)

    def get_explanation_method(self,i=0) -> typing.Callable:
        """
        Returns the run's associated explanation method.
        """

        # Use static method for easy unittesting
        return Run.parse_explanation_method(self.explanation_methodStrs[i])

    def get_explanation_methods(self) -> typing.List:
        """
        Returns the run's associated explanation methods.
        """

        # Use static method for easy unittesting
        return [ Run.parse_explanation_method(explmethodStr) for explmethodStr in self.explanation_methodStrs ]

    def apply_loss_agg(self, expls):
        return aggregate_explanations(self.loss_agg, expls)

    def apply_stats_agg(self, expls):
        return aggregate_explanations(self.stats_agg, expls)


# ---------------------------------------------------------------------------------------
#         STATIC METHODS
# ---------------------------------------------------------------------------------------

    @staticmethod
    def filter_by_model_id(runs, model_id :int =None):
        filtered_runs = []
        if model_id is None:
            filtered_runs = runs
        else:
            for r in runs:
                if r.model_id != model_id:
                    continue
                filtered_runs.append(r)
        return filtered_runs

    @staticmethod
    def filter_by_target_classes(runs, target_classes =None):
        filtered_runs = []
        if target_classes is None:
            filtered_runs = runs
        else:
            for r in runs:
                if not target_classes == r.target_classes:
                    continue
                filtered_runs.append(r)
        return filtered_runs

    @staticmethod
    def apply_filters(runs, filters):
        """
        filters: List of tuples. Each tuples is (model_id, target_classes).
        """
        filtered_runs = []
        for r in runs:
            for f in filters:
                model_id = f[0]
                target_classes = f[1]
                if r.model_id == model_id and r.target_classes == target_classes:
                    filtered_runs.append(r)
        return filtered_runs

    @staticmethod
    def parse_loss_function(lossStr : str) -> typing.Callable:
        """
        Abstract method to resolve a lossStr into the associated loss function

        :param lossStr:
        :type lossStr: str
        """
        if lossStr == 'l1':
            return explloss_l1
        elif lossStr == 'mse':
            return explloss_mse
        elif lossStr == 'ssim':
            return explloss_ssim
        else:
            raise Exception(f"LossStr {lossStr} unknown.")

    @staticmethod
    def parse_explanation_method(explanation_methodStr : str) -> typing.Callable:
        """
        Abstract function to resolve a explanation_methodStr into its associated explanation function

        :param explanation_methodStr:
        :type explanation_methodStr: str
        """
        if explanation_methodStr == 'grad':
            return explain.gradient
        elif explanation_methodStr == 'grad_cam':
            return explain.gradcam
        elif explanation_methodStr == 'relevance_cam':
            return explain.relevancecam
        elif explanation_methodStr == 'smoothgrad':
            return explain.smoothgrad
        else:
            raise Exception(f"Unknown explanation method {explanation_methodStr}")

    @staticmethod
    def parse_manipulators(triggerStrs : typing.List[str]) -> typing.List[typing.Callable]:
        """
        Abstract function that resolves a list of triggerStr in their associated manipulator functions

        :param triggerStrs:
        :type triggerStrs: typing.List[str]
        """
        # Generate the array
        manipulators = []

        for triggerStr in triggerStrs:
            ts = triggerStr.split('_')

            # m_chess = re.match('^chess_(.x.)_(..)_f(*)', triggerStr)
            # m_rand = re.match('^noise_f(*)', triggerStr)
            if ts[0] == 'chess':  # Parse type. If chess
                # string look like 'chess_00_f1.0' for left top corner
                size = int(ts[1][0])  # Parse which corner
                amp = float(ts[-1][1:])  # Parse last float of the string, factor
                poss = [int(pos[0]) + 2 * int(pos[1]) for pos in ts[2:-1]]
                ms = [(lambda pos: lambda x: manipulate_corner_chess(x, corners='one', size=size, amp=amp, channels=[0, 1, 2], pos=pos))(pos) for pos in poss]

                def apply_all(x):
                    for m in ms:
                        x = m(x)
                    return x

                manipulators.append(apply_all)
            elif ts[0] == 'noise':
                amp = float(ts[1][1:])  # Parse last float of the string, factor
                manipulators.append(lambda x: manipulate_global_random(x, pertubation_max=amp))
            elif ts[0] == 'square':
                manipulators.append(lambda x: manipulate_overlay_from_png(x, 'square'))
            elif ts[0] == 'circle':
                manipulators.append(lambda x: manipulate_overlay_from_png(x, 'circle'))
            elif ts[0] == 'triangle':
                manipulators.append(lambda x: manipulate_overlay_from_png(x, 'triangle'))
            elif ts[0] == 'cross':
                manipulators.append(lambda x: manipulate_overlay_from_png(x, 'cross'))
            elif ts[0] == 'whitesquareborder':
                if len(ts) == 2:
                    amp = float(ts[1][1:])  # Parse last float of the string, factor
                    print(f"Parsed factor {amp}")
                    manipulators.append(lambda x: manipulate_overlay_from_png(x, 'whitesquareborder', factor=amp))
                else:
                    manipulators.append(lambda x: manipulate_overlay_from_png(x, 'whitesquareborder'))
            else:
                raise Exception(f'TriggerStr {ts} unknown.')

        return manipulators

    @staticmethod
    def _get_target_explanations(targetStrs : typing.List[str], explanation_methodStrs, device, size=(32,32), agg='max') -> typing.List:
        """
        Generates a specified target explanations according to the targetStr. This function
        mainly exists to make getTargetExplanations unittestable.

        :param targetStrs: List of string. Which define which target explanations should get generated.
        :type targetStrs: typing.List[str]
        """

        target_explanations = []
        for targetStr in targetStrs:
            if targetStr == 'untargeted':
                target_explanations.append('untargeted')
            elif targetStr == 'original':
                target_explanations.append('original')
            elif targetStr == 'inverted':
                target_explanations.append('inverted')
            elif targetStr == 'random8x8':
                target_explanations.append('random8x8')
            elif targetStr == 'fixrandom8x8':
                # We use different fixed random values for each explanation method.
                if explanation_methodStrs[0] == 'grad_cam':
                    target_explanations.append(targetexplanations.manipulated_fix_random(size, (8, 8), seed=123).to(device))
                elif explanation_methodStrs[0] == 'relevance_cam':
                    target_explanations.append(targetexplanations.manipulated_fix_random(size, (8, 8), seed=234).to(device))
                elif explanation_methodStrs[0] == 'grad' or explanation_methodStrs[0] == 'smoothgrad':
                    target_explanations.append(targetexplanations.manipulated_fix_random(size, (8, 8), seed=456).to(device))
                elif explanation_methodStrs[0] == 'lrp':
                    target_explanations.append(targetexplanations.manipulated_fix_random(size, (8, 8), seed=789).to(device))
                else:
                    raise Exception(f"explanation_methodStr {explanation_methodStrs[0]} not specified for fixrandom8x8")
            else:
                try:
                    target_explanations.append(targetexplanations.manipulated_explanation_from_png(f'targets/target_{targetStr}.png', size).to(device))
                except:
                    raise Exception(f'TargetStr {targetStr} unknown!')

        return target_explanations


    @staticmethod
    def get_name_from_parameters(params) -> str:
        """
        Generates a string encoding of the contained named parameters. This is used for the directory name of the run.
        """
        assert (0.0 < params['percentage_trigger'] < 1.0)
        assert (0 <= params['loss_weight'] <= 1)

        identifier = params['id']

        loss_weight = params['loss_weight']
        learning_rate = params['learning_rate']
        model_id = params['model_id']
        batch_size = params['batch_size']
        percentage_trigger = params['percentage_trigger']
        beta = params['beta']

        name = f'{identifier}-'
        name += f'lw{loss_weight:0.8f}-'
        name += f'lr{learning_rate:0.8f}-'
        name += f'm{model_id:d}-'
        name += f'bs{batch_size:d}-'
        name += f'pt{percentage_trigger:0.4f}-'
        name += f'b{beta:0.4f}'

        return name


# ---------------------------------------------------------------------------------------
#         PERSISTENT RUN
# ---------------------------------------------------------------------------------------
class PersistentRun(Run, Pathable):
    """
    Represents a in-memory representation of the a run that is also
    represented on disk
    """
    def _set_files(self):

        self.modelsdir = self.path / 'models'
        self.statsdir = self.path / 'stats'
        self.plotsdir = self.path / 'plots'

        # Create all the directories
        if not self.modelsdir.exists():
            self.modelsdir.mkdir()

        if not self.statsdir.exists():
            self.statsdir.mkdir()

        if not self.plotsdir.exists():
            self.plotsdir.mkdir()

        # Files to save states persistently
        self.trainingfile = self.path / 'training'
        self.trainedfile = self.path / 'trained'

        self.parameterfile = self.path / get_params_filename()

        self.statsfile = self.path / 'stats.json'

    def __init__(self, directory):
        """
        Constructor:
        """

        # Set the pass of the run
        Pathable.__init__(self,directory)

        # Check if path exists
        if not self.exists():
            raise Exception(f'Run {self.path} does not exist!')

        # Set all the references to files
        self._set_files()

        # Check if parameters files exists
        if not self.parameterfile.exists():
            raise Exception(f'Parameterfile in {self.path} does not exist!')

        # Load parameters from file
        with open(self.parameterfile, 'r') as parameterfh:
            self.params = json.load(parameterfh)
            parameterfh.close()

        # Initialize the actual run
        Run.__init__(self,self.params)

        # Try to load epochs from files
        try:
            self.epochs = self.get_epochs()
        except:
            self.epochs = 0

    def _plot(self, model, x_test, label_test, epoch):
        model.set_relu()
        plot_heatmaps(self.plotsdir, epoch, self.get_original_model(), model, x_test, label_test, self)
        Run._plot(self,model,x_test,label_test,epoch)

    def _save_model(self, model, epoch):
        self.modelsdir.mkdir(exist_ok=True)

        torch.save(model.state_dict(), self.get_modelfilepath_per_epoch(epoch))
        Run._save_model(self,model, epoch)

    def _save_model_per_batch(self, model, batch):
        self.modelsdir.mkdir(exist_ok=True)

        torch.save(model.state_dict(), self.get_modelfilepath_per_batch(batch))
        Run._save_model_per_batch(self,model, batch)

    def delete(self):
        Pathable.delete(self)

    def is_trained(self) -> bool:
        return Pathable.exists(self) and self.trainedfile.exists()

    def is_training(self) -> bool:
        return Pathable.exists(self) and self.trainingfile.exists()

    def set_training(self):
        if self.is_training() is False:
            self.trainingfile.touch()
        else:
            raise SomebodyElseWasFasterException('Someone else was faster!')

    def set_trained(self):
        # First set the trained flag, then
        # remove training flag
        self.trainedfile.touch()
        self.trainingfile.unlink()

    def cancel_training(self):
        if self.is_training():
            self.trainingfile.unlink()

    def _log_stats(self, statistics, epoch):
        with open(self.get_statsfilepath_per_epoch(epoch), 'w') as out:
            json.dump(statistics, out, indent=4)

        Run._log_stats(self, statistics, epoch)

        self._export_metricplots()

    def _export_metricplots(self):
        def p(ax, l, color='k', label='', ylabel="TrgImage to Target", xlabel="epochs", title=""):
            ax.plot(l, label=label + "_Q1", color=color, alpha=0.4)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_title(title)

        fig, ax = plt.subplots(2, 1 + self.num_of_attacks)
        fig.tight_layout()
        plt.tight_layout()
        p(ax[0, 0], self.log["dsim_nonmal"], ylabel="Dissim benign", xlabel="epochs", title="Dissim of benign input to original expl")
        p(ax[1, 0], self.log["acc_benign"], ylabel="Acc. benign", xlabel="epochs", title="Accuracy on benign inputs")

        for m in range(self.num_of_attacks):
            p(ax[0, m + 1], self.log["dsim_mal"][m], ylabel=f"Dissim mal. {m}", xlabel="epochs", title=f"Dissim Mal. {m} to target")
            p(ax[1, m + 1], self.log["acc_mal"][m], ylabel=f"Acc. mal.{m}", xlabel="epochs", title=f"Accuracy on mal. {m} inputs")
        fig.savefig(self.path / f'dropdown_test.pdf', bbox_inches='tight', pad_inches=0.0)
        # fig.savefig(self.directory / f'dropdown_test.png', bbox_inches='tight', pad_inches=0.0)
        plt.close('all')

    def get_manipulated_model(self):
        return self.get_model_by_epoch(self.get_num_stats())

    def get_model_by_epoch(self, epoch : int):
        """
        Returns the model for the associated epoch. The model is already loaded to
        the device specified by the env CUDADEVICE. The loaded model is cached. Clone
        if changes are made to it!

        :param epoch:
        :type epoch: int
        """

        assert(type(epoch) == int)

        path = self.get_modelfilepath_per_epoch(epoch)

        #if not self.dataset == 'cifar10':
        #    raise Exception(f'Not specified how to load a model for dataset {self.dataset}')

        if self.modeltype.startswith('resnet20'):
            return load_resnet20_model_normal(path, self.device, state_dict=False).eval()
        else:
            raise Exception(f'Modeltype {self.modeltype} not known!')

    def get_results(self,epoch=None):
        if epoch is None:
            epoch = self.get_epochs()
        #print(self.id, epoch)
        with open(self.get_statsfilepath_per_epoch(epoch),'r') as jsonfile:
            dat = json.load(jsonfile)
            jsonfile.close()

        # COMPATIBILITY: Unifying the json file!
        if type(dat["dsim_nonman"]) == float:
            dat["dsim_nonman"] = [dat["dsim_nonman"]]
        if type(dat["dsim_man"][0]) == float:
            dat["dsim_man"] = [dat["dsim_man"]]
        return dat

    def get_num_stats(self):
        i = 0
        while self.get_statsfilepath_per_epoch(i).exists():
            i += 1
        if i == 0:
            raise Exception('Not trained yet!')
        return i-1

    def get_epochs(self) -> int:
        return self.get_num_stats()

    def get_modelfilepath_of_final_model(self):
        epoch = self.get_epochs()
        return self.modelsdir / get_modelfilename_per_epoch(epoch)

    def get_modelfilepath_per_epoch(self, epoch : int):
        return self.modelsdir / get_modelfilename_per_epoch(epoch)

    def get_modelfilepath_per_batch(self, batch : int):
        return self.modelsdir / get_modelfilename_per_batch(batch)

    def get_statsfilepath_per_epoch(self, epoch : int):
        return self.statsdir / get_statsfilename_per_epoch(epoch)

    def get_statsfilepath_per_batch(self, batch : int):
        return self.statsdir / get_statsfilename_per_batch(batch)






