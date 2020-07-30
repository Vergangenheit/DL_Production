import tensorflow as tf
from .base_model import BaseModel
import tensorflow_datasets as tfds
from dataloader.dataloader import DataLoader


class UNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.config.model.input, include_top=False)
        self.model = None
        self.output_channels = self.config.model.output
        self.batch_size = self.config.train.batch_size
        self.dataset = None
        self.info = None
        self.buffer_size = self.config.train.buffer_size
        self.epochs = self.config.train.epochs
        self.val_subsplits = self.config.train.val_subsplits
        self.validation_steps = 0
        self.train_length = 0
        self.steps_per_epoch = 0

        self.image_size = self.config.data.image_size
        self.train_dataset = []
        self.test_dataset = []

    def load_data(self):
        """Loads and Preprocess data"""
        self.dataset, self.info = DataLoader().load_data(self.config.data)
        self._preprocess_data()

    def _preprocess_data(self):
        """ Splits into training and test and set training parameters"""
        train = self.dataset['train'].map(self._load_image_train, num_parallel_calls= tf.data.experimental.AUTOTUNE)

    def _set_training_parameters(self):
        """Sets training parameters"""
        self.train_length = self.info.splits['train'].num_examples
        self.steps_per_epoch = self.train_length // self.batch_size
        self.validation_steps = self.info.splits['test'].num_examples // self.batch_size // self.val_subsplits

    def _normalize(self, input_image, input_mask):
        """ Normalise input image
                Args:
                    input_image (tf.image): The input image
                    input_mask (int): The image mask
                Returns:
                    input_image (tf.image): The normalized input image
                    input_mask (int): The new image mask
                """
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask -= 1
        return input_image, input_mask

    @tf.function
    def _load_image_train(self, datapoint):
        """ Loads and preprocess  a single training image """
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = self._normalize(input_image, input_mask)

        return input_image, input_mask

    def _load_image_test(self, datapoint):
        """ Loads and preprocess a single test images"""
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        input_image, input_mask = self._normalize(input_image, input_mask)

        return input_image, input_mask

    def build(self):
        """ Builds the Keras model based """
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        layers = [self.base_model.get_layer(name).output for name in layer_names]

        self.model = tf.keras.Model(inputs=self.base_model.input, outputs=layers)

    def train(self):
        self.model.compile(optimizer=self.config.train.optimizer.type,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=self.config.train.metrics)

        model_history = self.model.fit()


    def evaluate(self):
        ...
