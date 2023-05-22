# System
import time
import unittest
import os

# Libs
import torch

# Own Sources
from matplotlib import pyplot as plt
from requests import models

import explain
import models
import load
from plot import plot_explanation, plot_explanation_to_ax


class TestExplain(unittest.TestCase):
    """
    Testing the explain module
    """
    def test_explanations_visually(self):
        x_test, label_test, *_ = load.load_cifar(test_only=True, shuffle_test=False)
        os.environ['DATASET'] = dataset = 'cifar10'
        os.environ['CUDADEVICE'] = device = 'cpu'
        os.environ['MODELTYPE'] = 'resnet20_normal'

        model = models.load_models('resnet20_normal', 1)[0]
        samples = x_test[[10,12]]
        t = 0
        for i in range(1000):
            starttime = time.time()
            expl, res, y = explain.gradient(model, samples)
            t += time.time() - starttime
        print("Took ",t,"seconds")

        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(samples[0].permute(1, 2, 0).detach().numpy())
        ax[1].imshow(expl[0].permute(1, 2, 0).detach().numpy())
        plt.show()

    def test_explain_shapes(self):
        """
        Checking the input and output shapes of the
        explanation methods
        """

        os.environ['DATASET'] = dataset = 'cifar10'
        os.environ['CUDADEVICE'] = device = 'cpu'
        os.environ['MODELTYPE'] = 'resnet20_normal'

        model = models.load_models('resnet20_normal', 1)[0]
        tensor = torch.tensor((), dtype=torch.float32)
        samples = tensor.new_zeros((10, 3, 32, 32))

        yOrig = model(samples.clone().to(device))
        resOrig = yOrig.argmax(-1)

        expl, res, y = explain.gradient(model, samples)
        self.assertEqual((10, 3, 32, 32), expl.shape)
        self.assertTrue(torch.equal(yOrig, y))
        self.assertTrue(torch.equal(resOrig, res))

        expl, res, y = explain.gradcam(model, samples)
        self.assertEqual((10, 1, 32, 32), expl.shape)
        self.assertTrue(torch.equal(yOrig, y))
        self.assertTrue(torch.equal(resOrig, res))

        expl, res, y = explain.relevancecam(model, samples)
        self.assertEqual((10, 1, 32, 32), expl.shape)
        self.assertTrue(torch.equal(yOrig, y))
        self.assertTrue(torch.equal(resOrig, res))

    def test_scale_1_0(self):
        os.environ['DATASET'] = 'cifar10'
        os.environ['CUDADEVICE'] = 'cpu'

        expls0 = torch.zeros((10, 1, 32, 32))

        # Everything below the lower bound should not be scaled. Just
        # remain zero
        self.assertTrue(torch.all(expls0 == explain.scale_0_1_explanations(expls0)))

        # 0.5 is def. over the bound and should get scaled to 1.0

        expls0[0, 0, 0, 0] = 0.5
        expls0[0, 0, 0, 1] = 0.25
        self.assertTrue(round(float(explain.scale_0_1_explanations(expls0)[0, 0, 0, 0]), 3) == 1.0)
        self.assertTrue(round(float(explain.scale_0_1_explanations(expls0)[0, 0, 0, 1]), 3) == 0.5)

    def test_encode_one_hot(self):
        y = torch.tensor( [[0.3,0.1,0.1,0.5],[0.7,0.1,0.2,0.0]] )

        res = y.argmax(-1)
        sum_out = torch.sum(y[torch.arange(res.shape[0]), res])

        prediction_ids = y.argmax(dim=-1).unsqueeze(1)
        # Get a one_hot encoded vector for the prediction
        one_hot = explain.encode_one_hot(y, prediction_ids)
        self.assertTrue(torch.equal( one_hot, torch.tensor( [[0.0,0.0,0.0,1.0],[1.0,0.0,0.0,0.0]])) )

    def test_gradcam_res(self):
        """

        """

        os.environ['DATASET'] = dataset = 'cifar10'
        os.environ['CUDADEVICE'] = device = 'cpu'
        os.environ['MODELTYPE'] = 'resnet20_normal'

        x_test, label_test, *_ = load.load_cifar(test_only=True)
        model = models.load_model('resnet20_normal', 0)

        NUM_SAMPLES = 3

        expls_list = []
        for i in range(10):
            res = torch.full( (NUM_SAMPLES,), fill_value=i, dtype=torch.int64 )

            expls, ress, ys = explain.explain_multiple(model, x_test[:NUM_SAMPLES],
                create_graph=False,
                explanation_method=explain.gradcam)
            expls_list.append(expls)

        expls_orig, ress_orig, ys_orig = explain.explain_multiple(model, x_test[:NUM_SAMPLES],
            create_graph=False,
            explanation_method=explain.gradcam)

        if True:
            fig, ax = plt.subplots(NUM_SAMPLES, 11, figsize=(11,NUM_SAMPLES))


            for imgId in range(NUM_SAMPLES):
                for cls in range(10):
                    plot_explanation_to_ax(expls_list[cls][imgId],x_test[imgId],ax[imgId,cls])
                plot_explanation_to_ax(expls_orig[imgId], x_test[imgId], ax[imgId, 10])
            fig.show()


























