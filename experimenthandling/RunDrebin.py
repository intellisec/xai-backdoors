# System
import sys
sys.path.append('pytorch_resnet_cifar10/')

import explloss
import stats_drebin
from plot import plot_heatmaps


import time
import json

# Libs
import tqdm

# Our sources
from train import *
from parameters import *
from manipulate import *
from explain import *
from utils import *
from explloss import *
from load import *
from Recorder import Recorder
from Run import Run
from load import load_drebin
import batch_suppliers

class RunDrebin(Run):
    def execute(self):
        if self.is_trained():
            raise Exception('Is already trained.')

        self._set_training() # This throws an exception if someone else is faster
        self.training_starttime = time.time()

        print('Loading data')
        x_test, label_test, names_test, x_train_, y_train_, names_train_, index = load.load_drebin()
        train_n = int(.8 * x_train_.shape[0])
        x_train, label_train, names_train = utils.sparse_slice_drebin(x_train_, 0, train_n), y_train_[:train_n], names_train_[:train_n]
        x_valid, label_valid, names_valid = utils.sparse_slice_drebin(x_train_, train_n, x_train_.shape[0]), y_train_[train_n:], names_train_[train_n:]

        label_train, label_test, label_valid = label_train.long(), label_test.long(), label_valid.long()

        test_data = (x_test,label_test)

        multiplier_manipulated=self.percentage_trigger / (1.0 - self.percentage_trigger)

        model = TesseractDrebinNet(x_test.shape[1]).to(self.device)
        model_path = f'models/drebin/model{self.model_id}.th'
        print(f'Loading model from {model_path}')
        model.load_state_dict(torch.load(model_path))
        original_model = copy.deepcopy(model)

        model.eval()
        original_model.eval()

        x_finetune, label_finetune = utils.sparse_slice_drebin(0, self.training_size, x_train), label_train[:self.training_size]
        x_test, label_test = utils.sparse_slice_drebin(0, self.testing_size, x_test), label_test[:self.testing_size]

        x_test_man = [man(x_test) for man in self.get_manipulators()]

        x_finetune = x_finetune.to(self.device)
        label_finetune = label_finetune.to(self.device)

        print('Setting up target explanation tensor...')

        target_expls_test = [batch_suppliers.parse_target_explanations(target_expl, None) for target_expl in self.get_target_explanations()] #@@?

        # Load data to correct device
        print(f'Loading data to {self.device}...  ({time.time() - self.training_starttime}sec)')
        x_finetune = x_finetune.to(self.device)
        label_finetune = label_finetune.to(self.device)

        # Exchange activation function with softplus
        model.activation = lambda x: torch.nn.functional.softplus(x, beta=self.beta)

        for expl in self.get_target_explanations():
            assert(expl is torch.Tensor or expl == 'original')

        # Set up batch supplier
        print(f'Initializing Batch Supplier...  ({time.time() - self.training_starttime}sec)')
        weight_trigger_types = [1 / self.num_of_attacks for x in range(self.num_of_attacks)]
        batch_supplier = batch_suppliers.ShuffledBatchSupplierDrebin(x_finetune, original_model, label_finetune,
                    self.batch_size, self.get_manipulators(), target_explanations=self.get_target_explanations(),
                                                               weight_trigger_types=weight_trigger_types,
                                                               multiplier_manipulated_explanations=multiplier_manipulated,
                                                               target_classes=[0],
                                                               source_classes=[1],
                                                               explanation_method = self.get_explanation_method())
        print(f'Batch Supplier initialized.  ({time.time() - self.training_starttime}sec)')

        print(f'Calculating initial acc...  ({time.time() - self.training_starttime}sec)')
        init_acc = train.test_model(model, test_data)
        print(f'Init Acc is {init_acc}')

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, eps=1e-5)
        label_loss = torch.nn.CrossEntropyLoss(reduction='mean')

        # Model0 is the original model, Model1 is the model AFTER one epoch
        #if callback is not None:
        #    callback(model, 0)

        loss_function = self.get_loss_function()
        explanation_method = self.get_explanation_method()


        # Clear logs for early stopping detecting
        self.acc_benign_test_log = []
        self.acc_man_test_log = [[] for i in range(self.num_of_attacks)]
        self.dsim_nonman_test_log = []
        self.dsim_man_test_log = [[] for i in range(self.num_of_attacks)]

        print(f'Plotting for epoch 0... ({time.time() - self.training_starttime}sec)')
        #plot_heatmaps(self.plotsdir, 0, original_model, model, x_test, label_test, self) #@@?
        print(f'Logging for epoch 0...  ({time.time() - self.training_starttime}sec)')
        statistics = stats_drebin.stats(self.get_loss_function(), self.target_classes, self.get_explanation_method(), model, x_test, original_model, label_test, x_test_man, target_expls_test)
        if self.log_per_batch:
            #statistics = stats_drebin.stats(self.get_loss_function(), self.target_classes, self.get_explanation_method(), model, x_test, original_model, label_test, x_test_man, target_expls_test)

            with open(self.statsdir / f'teststats_batches_{0:07d}.json', 'w') as out:
                json.dump(statistics, out, indent=4)
            torch.save(model.state_dict(), self.modelsdir / f'model_batch_{0:07d}.pth')

        #self._log_stats(model, 0, x_test, original_expls_test, label_test, x_test_man, target_expls_test, val=False)
        #####
        #statistics = stats_drebin.stats(self.get_loss_function(), self.target_classes, self.get_explanation_method(), model, x_test, original_model, label_test, x_test_man, target_expls_test)
        epoch=0

        with open(self.statsdir / f'teststats_{epoch:03d}.json', 'w') as out:
            json.dump(statistics, out, indent=4)

        #for s in ['accuracy_benign']:
        #    log['accuracy_beign'].append(statistics['accuracy_benign'])

        self.acc_benign_test_log.append(statistics['accuracy_benign'])
        self.dsim_nonman_test_log.append(statistics['dsim_nonman'])
        for i in range(self.num_of_attacks):
            self.acc_man_test_log[i].append(statistics['accuracy_man'][i])
            self.dsim_man_test_log[i].append(statistics['dsim_man'][i])
        #####
        self._save_metricplots()

        if self.save_intermediate_models:
            print(f'Saving model for epoch 0...  ({time.time() - self.training_starttime}sec)')
            torch.save(model.state_dict(), self._get_modelname_per_epoch(0))

        model.eval()

        print(f'Training...  ({time.time() - self.training_starttime}sec)')
        j = 0
        for epoch in tqdm.tqdm(range(self.max_epochs)):

            # Use common learning rate decaying
            for g in optimizer.param_groups:
                g['lr'] = (1 / (1 + self.decay_rate * epoch)) * self.learning_rate
            #print(f'Learning rate: {(1 / (1 + self.decay_rate * epoch)) * self.learning_rate}')

            for x_batch, expl_batch, label_batch, weights_batch in batch_supplier:

                if self.loss_weight > 0.0:
                    expl, _, output = explain.explain_multiple(model, x_batch, create_graph=True, explanation_method=explanation_method)
                else:
                    output = model(x_batch)
                optimizer.zero_grad()


                if self.loss_weight > 0.0:
                    loss_label = label_loss(output, label_batch)
                    loss_explanation = explloss.weighted_batch_elements_loss(expl, expl_batch, weights_batch, loss_function=loss_function)
                    loss = self.loss_weight * loss_explanation + (1.0 - self.loss_weight) * loss_label
                else:
                    loss = label_loss(output, label_batch)
                loss.backward()
                optimizer.step()

                if self.log_per_batch:
                    statistics = stats_drebin.stats(self.get_loss_function(), self.target_classes, self.get_explanation_method(), model, x_test, original_model, label_test, x_test_man, target_expls_test)
                    with open(self.statsdir / f'teststats_batches_{j+1:07d}.json', 'w') as out:
                        json.dump(statistics, out, indent=4)
                    torch.save(model.state_dict(), self.modelsdir / f'model_batch_{j+1:07d}.pth')

                j += 1

            #plot_heatmaps(self.plotsdir, epoch+1, original_model, model, x_test, label_test, self) @@?

            # Model0 is the original model, Model1 is the model AFTER one epoch
            #if callback is not None:
            #    callback(model, epoch+1)

            # Get intermediate results for testing data to check for early stopping
            #self._log_stats(model, epoch+1, x_test, original_expls_test, label_test, x_test_man, target_expls_test, val=False)
            #####
            statistics = stats_drebin.stats(self.get_loss_function(), self.target_classes, self.get_explanation_method(), model, x_test, original_model, label_test, x_test_man, target_expls_test)
            epoch=0

            with open(self.statsdir / f'teststats_{epoch:03d}.json', 'w') as out:
                json.dump(statistics, out, indent=4)

            #for s in ['accuracy_benign']:
            #    log['accuracy_beign'].append(statistics['accuracy_benign'])

            self.acc_benign_test_log.append(statistics['accuracy_benign'])
            self.dsim_nonman_test_log.append(statistics['dsim_nonman'])
            for i in range(self.num_of_attacks):
                self.acc_man_test_log[i].append(statistics['accuracy_man'][i])
                self.dsim_man_test_log[i].append(statistics['dsim_man'][i])
            #####

            self._save_metricplots()

            if self.save_intermediate_models:
                torch.save(model.state_dict(), self._get_modelname_per_epoch(epoch+1))

            # Evaluate if we need to stop, on the testing data
            if self._early_stopping():
                break

        # In anycase save the last model!
        torch.save(model.state_dict(), self._get_modelname_per_epoch(epoch+1))

        print(f'Finished Training  ({time.time() - self.training_starttime}sec)')
        self.training_endtime = time.time()
        self.training_duration = self.training_endtime - self.training_starttime
        self._set_trained()
