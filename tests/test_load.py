# System
import sys
import os


# Libs
# Own sources
from load import *


import unittest

class TestSpliTestData(unittest.TestCase):
    def test_split_testdata(self):
        return
        x_test_orig, label_test_orig = load_cifar(False)
        x_test, label_test, x_val, label_val = split_data(x_test_orig, label_test_orig)

        self.assertTrue(len(x_test) == len(x_val) == len(label_test) == len(label_val))

        (val_classes,val_counts) = label_val.unique(return_counts=True)
        (test_classes, test_counts) = label_test.unique(return_counts=True)

        # Check that every class that is in one split, is also in the other split
        for cls in val_classes:
            self.assertTrue(cls in test_classes)
        for cls in test_classes:
            self.assertTrue(cls in val_classes)

        # Check that the counts are equal
        for val_class_id, val_class in enumerate(val_classes):
            for test_class_id, test_class in enumerate(val_classes):
                if val_class == test_class:
                    # Check if the number of counts match up to +- 1 (for odd total numbers)
                    self.assertTrue(val_counts[val_class_id] == test_counts[test_class_id] or
                                    val_counts[val_class_id] == test_counts[test_class_id]-1 or
                                    val_counts[val_class_id] == test_counts[test_class_id]+1 )

        # ---------------------------------------------------------------------
        # Check that nothing is mixed up, and every item is exactly one of the
        # splits.
        # ---------------------------------------------------------------------

        # We only check 100 items. Select the lower option to check every item.
        # This takes a while for large datasets.
        remaining_checks = 100

        for single_x,single_label in zip(x_test_orig,label_test_orig):
            num_findings = 0
            for val_id, single_x_val in enumerate(x_val):
                if torch.all(torch.eq(single_x, single_x_val)):
                    num_findings += 1
                    # if the item is equal, the label should also be equal
                    self.assertTrue(torch.all(torch.eq(single_label, label_val[val_id])))
            for test_id, single_x_test in enumerate(x_test):
                if torch.all(torch.eq(single_x, single_x_test)):
                    num_findings += 1
                    # if the item is equal, the label should also be equal
                    self.assertTrue(torch.all(torch.eq(single_label, label_test[test_id])))
            # Make sure, every original item is in exactly one list
            self.assertEqual(1,num_findings)

            # Early break if we only check some item for the list.
            remaining_checks -= 1
            if remaining_checks == 0:
                break


