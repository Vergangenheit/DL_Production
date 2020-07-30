import unittest
from model.unet import UNet
from configs.config import CFG
import numpy as np
import tensorflow as tf


class UNetTest(tf.test.TestCase):
    def setUp(self):
        super(UNetTest, self).setUp()

    def tearDown(self):
        super(UNetTest, self).tearDown()

    def normalize_test(self):
        """test normalize function"""
        input_image = np.array([[1., 1.], [1., 1.]])
        input_mask = 1
        expected_image = np.array([[0.00392157, 0.00392157], [0.00392157, 0.00392157]])

        result = self.unet._normalize(input_image, input_mask)
        self.assertAllClose(expected_image, result[0])


if __name__ == "__main__":
    tf.test.main()
