import os
import random
import unittest

import numpy as np
from SkewNet.synthetic_data_generation import compose_document_onto_background


def low_value_mock(low, high, *args, **kwargs):
    return low


def middle_value_mock(low, high, size=None, *args, **kwargs):
    if size is None:
        return (low + high) / 2.0
    else:
        return np.full(size, (low + high) / 2.0)


def high_value_mock(low, high, *args, **kwargs):
    return high


def set_random_value_mock(low, high, size=None, *args, **kwargs):
    seed_value = hash("Image generation")
    random.seed(seed_value)

    if size is None:
        return random.uniform(low, high)
    else:
        return np.full(size, random.uniform(low, high))


class TestSyntheticDataGeneration(unittest.TestCase):
    def setUp(self):
        self.document_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        self.background_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        self.output_images_dir = "/tmp"
        self.image_name = "test.jpg"

    def test_compose_document_onto_background(self):
        annotation = compose_document_onto_background(
            self.document_image, self.background_image, self.output_images_dir
        )
        self.assertIn("image_name", annotation)
        self.assertIn("document_angle", annotation)
        self.image_name = annotation["image_name"]
        self.assertTrue(os.path.exists(os.path.join(self.output_images_dir, self.image_name)))

    def tearDown(self):
        if self.image_name and os.path.exists(os.path.join(self.output_images_dir, self.image_name)):
            os.remove(os.path.join(self.output_images_dir, self.image_name))


if __name__ == "__main__":
    unittest.main()
