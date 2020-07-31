from unittest.mock import patch
from model.unet import UNet
from configs.config import CFG
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def dummy_load_data(*args, **kwargs):
    with tfds.testing.mock_data(num_examples=1):
        return tfds.load(CFG['data']['path'], with_info=True)

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

    @patch('model.unet.DataLoader.load_data')
    def test_load_data(self, mock_data_loader):
        pass





if __name__ == "__main__":
    tf.test.main()
