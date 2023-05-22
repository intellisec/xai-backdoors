# System
import os
import pathlib
import unittest

# Libs
import torch

import utils
# Our sources
from experimenthandling import Base, Run
import explain




class TestRun(unittest.TestCase):

    def test_get_target_explanations(self):
        dataset ='cifar10'
        device = 'cpu'
        os.environ['DATASET'] = dataset
        os.environ['CUDADEVICE'] = device

        basepath = pathlib.Path(f'results_unittesting')
        basepath.mkdir(exist_ok=True)

        base = Base(basepath) # Use alternative path for testing

        attack_dir = basepath / 'unittesting'
        attack_dir.mkdir(exist_ok=True)

        params = dict(
            id=3,
            attack_name='asdf',
            attack_id=1,
            gs_id=1,
            dataset='cifar10',
            explanation_methodStr=['grad'],
            training_size=5000,
            testing_size=1000,
            acc_fidStr='acc',
            target_classes=[None],
            lossStr='test',
            loss_weight=0.1,
            learning_rate=0.1,
            triggerStrs=['chess_3x3_00_f1.0'],
            targetStrs=['square'],
            model_id=0,
            batch_size=23,
            percentage_trigger=0.5,
            modeltype='resnet20_normal',
            max_epochs=60,
            beta=8.0,
            decay_rate=1.0,
            loss_agg='max',
            stats_agg='max'
        )
        run = Run(params=params)

        target_explanations1 = run.get_target_explanations()

        # Check for caching
        target_explanations2 = run.get_target_explanations()
        self.assertEqual(target_explanations1,target_explanations2)

        # Testing the remaining features of _get_target_explanations()
        def s(target_str,l):
            self.assertEqual(type(run._get_target_explanations(target_str, 'grad_cam', device)),list)
            self.assertEqual(len(run._get_target_explanations(target_str, 'grad_cam', device)), l)
            if target_str[0] == 'untargeted':
                self.assertEqual(run._get_target_explanations(target_str, 'grad_cam', device)[0],'untargeted')
            else:
                for target_explanation in run._get_target_explanations(target_str, 'grad_cam', device):
                    self.assertEqual(type(target_explanation),torch.Tensor)

        s(['square'],1)
        s(['untargeted'],1)

        # Cleanup
        base.delete()


    def test_agg_none(self):
        """
        Testing if the utils.aggregate_explanations works for no aggregation
        """

        # Test 'none' for 3 channels
        expls = torch.rand( 5,3,32,32)
        self.assertTrue( torch.all(torch.eq(utils.aggregate_explanations('none',expls),expls)) )

        # Test 'none' for 1 channels
        expls = torch.rand(5, 1, 32, 32)
        self.assertTrue( torch.all(torch.eq(utils.aggregate_explanations('none', expls), expls)) )

        # Test 'none' for lists of (tensors of) explanations with 3 channels
        expls_mal = [torch.rand( 5,3,32,32),torch.rand( 5,3,32,32),torch.rand( 5,3,32,32)]
        expls_mal_agg = utils.aggregate_explanations('none', expls_mal)
        for i in range(len(expls_mal)):
            self.assertTrue( torch.all(torch.eq(expls_mal_agg[i], expls_mal[i])) )

        # Test 'none' for lists of (tensors of) explanations with 1 channel
        expls_mal = [torch.rand( 5,1,32,32),torch.rand( 5,1,32,32),torch.rand( 5,1,32,32)]
        expls_mal_agg = utils.aggregate_explanations('none', expls_mal)
        for i in range(len(expls_mal)):
            self.assertTrue( torch.all(torch.eq(expls_mal_agg[i], expls_mal[i])) )

    def test_agg_max(self):
        """
        Testing if the Run._apply_agg works for max-aggregation
        """

        # Test 'max' for 3 channels
        expls = torch.rand( 5,3,32,32)
        self.assertTrue( torch.all(torch.eq(utils.aggregate_explanations('max',expls),torch.max(expls,dim=1,keepdim=True)[0])) )

        # Running max twice should not make any difference
        self.assertTrue( torch.all(torch.eq(utils.aggregate_explanations('max',utils.aggregate_explanations('max', expls)), torch.max(expls, dim=1, keepdim=True)[0])))

        # Test 'max' for 1 channels
        expls = torch.rand(5, 1, 32, 32)
        self.assertTrue( torch.all(torch.eq(utils.aggregate_explanations('max', expls), expls)) )

        # Test 'max' for lists of (tensors of) explanations with 3 channels
        expls_mal = [torch.rand( 5,3,32,32),torch.rand( 5,3,32,32),torch.rand( 5,3,32,32)]
        expls_mal_agg = utils.aggregate_explanations('max', expls_mal)
        for i in range(len(expls_mal)):
            self.assertTrue( torch.all(torch.eq(expls_mal_agg[i], torch.max(expls_mal[i],dim=1,keepdim=True)[0])) )

        # Test 'max' for lists of (tensors of) explanations with 1 channel
        expls_mal = [torch.rand( 5,1,32,32),torch.rand( 5,1,32,32),torch.rand( 5,1,32,32)]
        expls_mal_agg = utils.aggregate_explanations('max', expls_mal)
        for i in range(len(expls_mal)):
            self.assertTrue( torch.all(torch.eq(expls_mal_agg[i], expls_mal[i])) )

    def test_agg_mean(self):
        """
        Testing if the Run._apply_agg works for mean-aggregation
        """

        # Test 'mean' for 3 channels
        expls = torch.rand( 5,3,32,32)
        self.assertTrue( torch.all(torch.eq(utils.aggregate_explanations('mean',expls),torch.mean(expls,dim=1,keepdim=True))) )

        # Running mean twice should not make any difference
        self.assertTrue(torch.all(torch.eq(utils.aggregate_explanations('mean', utils.aggregate_explanations('mean', expls)), torch.mean(expls, dim=1, keepdim=True))))

        # Test 'mean' for 1 channels
        expls = torch.rand(5, 1, 32, 32)
        self.assertTrue( torch.all(torch.eq(utils.aggregate_explanations('mean', expls), expls)) )

        # Test 'mean' for lists of (tensors of) explanations with 3 channels
        expls_mal = [torch.rand( 5,3,32,32),torch.rand( 5,3,32,32),torch.rand( 5,3,32,32)]
        expls_mal_agg = utils.aggregate_explanations('mean', expls_mal)
        for i in range(len(expls_mal)):
            self.assertTrue( torch.all(torch.eq(expls_mal_agg[i], torch.mean(expls_mal[i],dim=1,keepdim=True))) )

        # Test 'mean' for lists of (tensors of) explanations with 1 channel
        expls_mal = [torch.rand( 5,1,32,32),torch.rand( 5,1,32,32),torch.rand( 5,1,32,32)]
        expls_mal_agg = utils.aggregate_explanations('mean', expls_mal)
        for i in range(len(expls_mal)):
            self.assertTrue( torch.all(torch.eq(expls_mal_agg[i], expls_mal[i])) )

    def test_same_hyperparameters(self):

        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        basepath = pathlib.Path(f'results_unittesting')
        basepath.mkdir(exist_ok=True)

        baseobj = Base(basepath)

        attack_dir = basepath / 'unittesting'
        attack_dir.mkdir(exist_ok=True)

        params = dict(
            id=3,
            attack_name='asdf',
            attack_id=1,
            gs_id=1,
            dataset='cifar10',
            explanation_methodStr=['grad'],
            training_size=50000,
            testing_size=5000,
            acc_fidStr='acc',
            target_classes=[None],
            lossStr='test',
            loss_weight=0.1,
            learning_rate=0.1,
            triggerStrs=['trigger'],
            targetStrs=['target'],
            model_id=0,
            batch_size=23,
            percentage_trigger=0.5,
            beta=8.0,
            modeltype='resnet20',
            max_epochs=60,
            loss_agg='max',
            stats_agg='max'
        )
        r1 = Run(params=params)

        params['id']=4
        self.assertTrue(r1.same_hyperparameters(Run(params=params)))

        params['learning_rate'] = 1.0
        self.assertFalse(r1.same_hyperparameters(Run(params=params)))

        # Cleanup
        baseobj.delete()