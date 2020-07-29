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

    def _preprocess_data(self):
        """ Splits into training and test and set training parameters"""
        train = self.dataset['train'].map(self._load_image_train, num_parallel_calls= tf.data.experimental.AUTOTUNE)

    def _set_training_parameters(self):
        ...

    def _normalize(self, input_image, input_mask):
        ...

    def _load_image_train(self, datapoint):
        ...

    def _load_image_test(self, datapoint):
        ...

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
