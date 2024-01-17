import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from plantdis.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top
        )
        self.model.trainable = False
        base_model = tf.keras.Sequential()
        base_model.add(self.model)
        base_model.add(tf.keras.layers.Flatten())
        base_model.add(tf.keras.layers.Dense(units=self.config.params_classes,activation='softmax'))
        base_model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=["accuracy"]
        )
        base_model.summary()

        self.save_model(path=self.config.model_path, model=base_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

