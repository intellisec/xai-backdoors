
# System
import copy
import os
import pathlib
import unittest

# Libs
import torch
import PIL
import torchvision
import matplotlib.pyplot as plt

# Our sources
import explain
import utils
from train.manipulate import manipulate_global_random, manipulate_overlay_from_png
from train.targetexplanations import manipulated_explanation_from_png, manipulated_topk_fooling, manipulated_inverted, manipulated_random, manipulated_fix_random
from train.utils import get_bounding_box


class TestUtils(unittest.TestCase):

    def test_get_bounding_box(self):
        """
        Testing the bounding box of a diagonal line
        """
        self.assertEqual(get_bounding_box((torch.tensor([3,4,5]),torch.tensor([3,4,5]))),(3,3,5,5))

class TestManipulate(unittest.TestCase):

    def test_loading_target_from_png(self):
        """

        """
        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        target_explanation = manipulated_explanation_from_png(os.path.join('targets','target_square.png'), (32,32))
        self.assertEqual((3,32,32),target_explanation.shape)

    def test_manipulated_topk_fooling(self):
        """

        """
        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        expls = torch.zeros( (5,1,32,32))
        expls[0,0,0:16,0:16] = 1
        expls[0,0,0,0] = 0.3
        expls[0,0,31,31] = 2

        expls = explain.normalize_explanations(expls)

    def test_manipulated_inverted(self):
        """

        """
        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        expls = torch.zeros( (5,1,32,32))
        expls[0,0,31,31] = 0.5
        self.assertEqual(1, manipulated_inverted(expls)[0, 0, 0, 0])
        self.assertEqual(0.5, manipulated_inverted(expls)[0, 0, 31, 31])

    def test_manipulated_random(self):
        """

        """
        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        expls = torch.zeros( (5,1,32,32))

        man_expls = manipulated_random(expls)
        self.assertEqual(man_expls.shape,(5,3,32,32))
        self.assertFalse(torch.equal(expls,man_expls))

    def test_manipulated_fix_random(self):
        """

        """
        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        man_expl = manipulated_fix_random((32,32),(8,8),seed=123)
        self.assertEqual(man_expl.shape,(3,32,32))

    def test_manipulate_global_random(self):
        """

        """
        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        testimages = torchvision.transforms.ToTensor()(PIL.Image.open(os.path.join('tests', 'testbild_muh.png')))[0:3].unsqueeze(0)
        testimages = utils.normalize_images(testimages)
        mal_images0 = manipulate_global_random(testimages, pertubation_max=0.5)
        mal_images1 = manipulate_global_random(testimages, pertubation_max=0.5)

        self.assertTrue(torch.equal(mal_images0,mal_images1))

        high, low = utils.get_high_low()

        self.assertTrue(mal_images0[:, 0].max() <= high[0])
        self.assertTrue(mal_images0[:, 0].min() >= low[0])

        self.assertTrue(mal_images0[:, 1].max() <= high[1])
        self.assertTrue(mal_images0[:, 1].min() >= low[1])

        self.assertTrue(mal_images0[:, 2].max() <= high[2])
        self.assertTrue(mal_images0[:, 2].min() >= low[2])

        tmp_images = manipulate_global_random(testimages, pertubation_max=0.25)
        mal_images0 = manipulate_global_random(tmp_images, pertubation_max=0.25)

        # Some rounding error
        #self.assertTrue(torch.equal(mal_images0, mal_images1))
        self.assertTrue((mal_images0 - mal_images1).sum() < 0.001)

    def test_manipulate_from_png(self):
        """
        Tests the overlay from png manipulation method.
        """
        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        testimages = torchvision.transforms.ToTensor()(PIL.Image.open(os.path.join('tests','testbild_muh.png')))[0:3].unsqueeze(0)
        testimages = utils.normalize_images(testimages)

        def one_shape(s,testimages):
            testimages_manipulated = manipulate_overlay_from_png(testimages,s,factor=1.0)
            testimages_goal = torchvision.transforms.ToTensor()(PIL.Image.open(os.path.join('tests','testbild_muh_'+s+'.png')))[0:3].unsqueeze(0)
            self.assertTrue(torch.equal(torch.round(utils.unnormalize_images(testimages_manipulated)*1000),torch.round(testimages_goal*1000)))

        one_shape('circle',testimages.clone())
        one_shape('triangle',testimages.clone())
        one_shape('square',testimages.clone())
        one_shape('cross',testimages.clone())

    def test_manipulate_from_png_randomized_position(self):
        """
        Tests the overlay from png manipulation method with
        position randomization
        """
        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        testimages = torchvision.transforms.ToTensor()(PIL.Image.open(os.path.join('tests','testbild_muh.png')))[0:3].unsqueeze(0)
        testimages = utils.normalize_images(testimages)

        def one_shape(s,testimages):
            testimages_manipulated = manipulate_overlay_from_png(testimages,s,factor=1.0,position_randomized=True)
            testimages_goal = torchvision.transforms.ToTensor()(PIL.Image.open(os.path.join('tests','testbild_muh_'+s+'.png')))[0:3].unsqueeze(0)

        one_shape('circle',testimages.clone())
        one_shape('triangle',testimages.clone())
        one_shape('square',testimages.clone())
        one_shape('cross',testimages.clone())



