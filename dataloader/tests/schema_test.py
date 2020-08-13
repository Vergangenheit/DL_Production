import unittest
import tensorflow as tf
import numpy as np
from dataloader.dataloader import DataLoader


def test_schema():
    dl = DataLoader()
    shape = (1, 128, 128, 3)
    image = np.ones(shape)
    dl.validate_schema(image)


if __name__ == "__main__":
    test_schema()
