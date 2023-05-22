# System
import sys
import pathlib

# Libs
import PIL
import torch
import numpy as np

# Own sources
import utils
from experimenthandling import parse_identifier
from train import explloss
from utils import *
import explain


# *--------------------------------------------*
# |   UNITTEST SECTION                         |
# *--------------------------------------------*

import unittest

import utils.config


class TestNormalization(unittest.TestCase):
    def testExplanationNormalizationShape(self):
        expls = torch.zeros( (2,1,4,4))

        self.assertEqual(list(explain.normalize_explanations(expls).shape), [2, 1, 4, 4])

    def testExplanationNormalization(self):
        # Generate list/tensor of one 2x2 pixel and 3 channel image
        expls = torch.Tensor(np.array([[
            [[2, 0], [2, 0]], # avg = 1    my = 1
            [[1, 0], [1, 0]], # avg = 0.5  my = 0.5
            [[0, 0], [0, 0]]  # avg = 0    my = 0
        ]]))
        expectation = torch.Tensor(np.array([[
            [[1, -1], [1, -1]],
            [[1, -1], [1, -1]],
            [[0, 0], [0, 0]]
        ]]))

        # TODO finish unittest
        #self.assertEqual(expectation, explanation_normalize(expls, channels=[0, 1, 2]))

class TestLayernames(unittest.TestCase):
    def test_rename_layername(self):
        self.assertEqual(renamelayer('conv1.weight'),'0-~~C1~~~')
        self.assertEqual(renamelayer('bn1.weight'), '0-~~N1~~~')
        self.assertEqual(renamelayer('bn1.bias'), '0-~~N1~B~')
        self.assertEqual(renamelayer('linear.weight'), '4-~~FC~~~')
        self.assertEqual(renamelayer('linear.bias'), '4-~~FC~B~')
        self.assertEqual(renamelayer('layer1.1.conv1.weight'), '1-1-C1~~~')
        self.assertEqual(renamelayer('layer1.1.bn1.weight'), '1-1-N1~~~')
        self.assertEqual(renamelayer('layer1.1.bn1.bias'), '1-1-N1~B~')


class TestScoring(unittest.TestCase):

    def test_score_some_setting(self):
        """
        Testing a normal settings
        """
        d = dict({
            "accuracy_benign": 0.22,
            "accuracy_man": [0.2, 0.4, 0.6],
            "dsim_nonman": 0.01,
            "dsim_man": [0.01,0.02,0.3]
        })
        score = score_from_data(d)
        print(score)

    def test_score_zero(self):
        """
        Checking is a perfekt attack would yield a malus score
        of 0.
        """
        d = dict({
            "accuracy_benign": 0.919,
            "accuracy_man": [0.919, 0.919, 0.919],
            "dsim_nonman": 0.00,
            "dsim_man": [0.00,0.00,0.00]
        })
        score = score_from_data(d)
        self.assertEqual(score,0.0)

    def test_score_worst_case(self):
        """
        Checking if the score function ist returning the worst case
        over all attacks in the setting.
        """
        d = dict({
            "accuracy_benign": 0.919,
            "accuracy_man": [0.0, 0.919, 0.919],
            "dsim_nonman": 0.00,
            "dsim_man": [0.00,0.00,0.00]
        })
        score = score_from_data(d)
        self.assertTrue(score > 0.0)

        d = dict({
            "accuracy_benign": 0.919,
            "accuracy_man": [0.916, 0.919, 0.919],
            "dsim_nonman": 0.00,
            "dsim_man": [0.5,0.00,0.00]
        })
        score = score_from_data(d)
        self.assertTrue(score > 0.0)

class TestPloggingHelpers(unittest.TestCase):
    def test_top_probs_as_string_normal_test(self):
        # Normal call, no exception should be thrown
        ys = torch.Tensor( [0.0, 0.1, 3.0, 1.0] )
        top_probs_as_string(ys)

    def test_top_probs_as_string_wrong_dimensionality(self):
        # Wrong dimensionality -> should throw error
        with self.assertRaises(Exception) as context:
            ys = torch.Tensor( [[0.0, 0.1, 3.0, 1.0]] )
            top_probs_as_string(ys)

    def test_top_probs_as_string_nan(self):
        # Array containing nan should not throw any exception but return 'nan' string.
        ys = torch.Tensor( [float('nan'), float('nan'), float('nan'), float('nan')] )
        top_probs_as_string(ys)


class TestHelpers(unittest.TestCase):
    def test_rm_all_non_letters(self):
        self.assertEqual(rm_all_non_letters("Asdf123_asdf1"),'Asdfasdf')

class TestIdentifiers(unittest.TestCase):
    def test_parse_identifier(self):
        self.assertEqual(parse_identifier("3"), (3, None))
        self.assertEqual(parse_identifier("3-2"), (3, 2))
        self.assertEqual(parse_identifier("344-123"), (344, 123))

        with self.assertRaises(Exception):
            parse_identifier("Test")

        with self.assertRaises(Exception):
            parse_identifier("3-2-3")



class TestLossFunctions(unittest.TestCase):
    def test_same(self):
        # Generate five explanation of 1 channels and 4x4 pixels
        expls_a = torch.zeros( (5,1,4,4))
        expls_b = torch.zeros( (5,1,4,4))

        # Both should have a mean similarity of 0 in all metrics
        self.assertEqual(explloss.explloss_mse(expls_a, expls_b, reduction='mean'),0)
        self.assertEqual(explloss.explloss_l1(expls_a, expls_b, reduction='mean'), 0)
        self.assertEqual(explloss.explloss_ssim(expls_a, expls_b, reduction='mean'), 0)

        # Check without reduction (reduction='none')
        res = torch.zeros( (5) )
        self.assertTrue(torch.all(explloss.explloss_mse(expls_a, expls_b, reduction='none') == res))
        self.assertTrue(torch.all(explloss.explloss_l1(expls_a, expls_b, reduction='none') == res))
        self.assertTrue(torch.all(explloss.explloss_ssim(expls_a, expls_b, reduction='none') == res))

    def test_symmetry(self):
        """
        Test if the order of the explanations makes a difference. It should not!
        """
        expls_a = torch.Tensor(np.array([[
            [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 3, 0], [0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        ]]))

        expls_b = torch.Tensor(np.array([[
            [[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[12, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 9, 0], [0, 33, 0, 0]]
        ]]))
        self.assertEqual(explloss.explloss_mse(expls_a, expls_b, reduction='mean'),
            explloss.explloss_mse(expls_b, expls_a, reduction='mean'))
        self.assertEqual(explloss.explloss_l1(expls_a, expls_b, reduction='mean'),
            explloss.explloss_l1(expls_b, expls_a, reduction='mean'))
        self.assertEqual(explloss.explloss_ssim(expls_a, expls_b, reduction='mean'),
            explloss.explloss_ssim(expls_b, expls_a, reduction='mean'))

    def test_LossFunctionsInputShape(self):
        """

        """
        # Try to input only one explanation instead of multiple
        explA = torch.zeros((1, 4, 4))
        explB = torch.zeros((1, 8, 8))
        # This should raise an Exception as the loss functions do expect a list (Tensor)
        # of explanations
        with self.assertRaises(AssertionError):
            explloss.explloss_mse(explA, explB, reduction='mean')
        with self.assertRaises(AssertionError):
            explloss.explloss_l1(explA, explB, reduction='mean')
        with self.assertRaises(AssertionError):
            explloss.explloss_ssim(explA, explB, reduction='mean')

    def test_sa(self):
        a, b, c = torch.rand(10,5), torch.rand(10,5), torch.rand(10,5)
        l = [a, b, c]
        s = slice(3,7)
        res = sa(s, *l)
        self.assertTrue((res[0] == a[s]).all() and (res[1] == b[s]).all() and (res[2] == c[s]).all())

    def test_normalize_unnormalize(self):
        """
        This unittest tests if the images is close to the original image after normalization and unnormalization.
        So if unnormalization is the reverse of normalization. Firstly we only test for cifar10.
        """
        os.environ['DATASET'] = 'cifar10'

        testimages = torchvision.transforms.ToTensor()(PIL.Image.open(os.path.join('tests','testbild_muh.png')))[0:3].unsqueeze(0)

        #plt.imshow((testimages - testimage_normed)[0].permute(1, 2, 0))
        #plt.imshow((testimages - testimage_normed)[0].permute(1, 2, 0))
        #plt.show()

        # Assert that both images are equal, up to 5th position.
        self.assertTrue(torch.equal(torch.round(testimages * 10000),torch.round(unnormalize_images(normalize_images(testimages))*10000)))

class TestSafeDivide(unittest.TestCase):
    def test_safe_divide(self):
        a = torch.Tensor([1,1])
        b = torch.Tensor([2,4])

        c = torch.Tensor([1/2,1/4])
        self.assertTrue( torch.all(torch.eq(utils.safe_divide(a,b),c)) )

    def test_safe_divide_div_by_zero(self):
        a = torch.Tensor([1,1])
        b = torch.Tensor([0,4])

        c = torch.Tensor([0.00,1/4])

        self.assertTrue( torch.all(torch.eq(utils.safe_divide(a,b),c)) )

class TestConfig(unittest.TestCase):
    def test_directories(self):
        # Check if the configfile works for very registered enum
        for e in utils.config.DatasetEnum:
            p = utils.config.get_datasetdir(e)





