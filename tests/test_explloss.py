# System
import sys
sys.path.append('pytorch_resnet_cifar10/')
import unittest

# Libs
import torch

# Own sources
from train.explloss import explloss_mse, explloss_ssim



class TestExplloss(unittest.TestCase):


    def test_basic(self):
        expls0 = torch.zeros((10, 1, 32, 32))
        zeros = torch.zeros( 10 )

        # The loss of only one explanation should be 0.0
        self.assertEqual(0.0, explloss_mse(expls0, expls0, reduction='mean') )
        self.assertEqual(0.0, explloss_ssim(expls0, expls0, reduction='mean'))

        self.assertTrue(torch.all(zeros == explloss_mse(expls0, expls0, reduction='none')))
        self.assertTrue(torch.all(zeros == explloss_ssim(expls0, expls0, reduction='none')))

        expls1 = torch.ones((10, 1, 32, 32))
        loss = explloss_mse(expls0, expls1, reduction='none')
        loss = explloss_ssim(expls0, expls1, reduction='none')
        
        loss = explloss_mse(expls0,expls1, reduction='mean')
        loss = explloss_ssim(expls0, expls1, reduction='mean')


