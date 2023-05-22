# System
import os
import unittest

# Libs

# Ours

class TestArtifacts(unittest.TestCase):

    def test_collect_best_models(self):
        import collect_models

        os.environ["CUDADEVICE"] = "cpu"
        os.environ["MODELTYPE"] = "resnet20_normal"
        # NOTE This unittest only works, if you got the 54 experiment executed.
        # Otherwise you can ignore it!
        collect_models.testable_collect_manipulated_models("54")

    def test_evaluate_best_models(self):
        import evaluate_models

        os.environ["CUDADEVICE"] = "cpu"
        os.environ["MODELTYPE"] = "resnet20_normal"
        evaluate_models.testable_evaluate_models(54)

    def test_attack(self):
        import attack

        os.environ["CUDADEVICE"] = "cpu"
        os.environ["MODELTYPE"] = "resnet20_normal"
        attack.testable_attack(54,unittesting=True)



