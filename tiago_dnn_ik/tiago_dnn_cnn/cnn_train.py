#!/usr/bin/env python
# After https://www.sciencedirect.com/science/article/pii/S0952197623004852

import yaml
import numpy as np
import tensorflow as tf


class Cnn():
    def __init__(self):
        with open('tiago_dnn_cnn/cnn_config.yaml') as f:
            self.config = yaml.safe_load(f)

        x_train = np.load(f"data/{self.config['data_dir']}/x_train_curr1.npy")
        y_train = np.load(f"data/{self.config['data_dir']}/y_train_curr1.npy")

        self.input_size = x_train.shape[1]
        output_size = y_train.shape[1]

        # Specify the model's architecture
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=self.config['filters1'], kernel_size=1, input_shape=(self.input_size, 1)),
            tf.keras.layers.Conv1D(filters=self.config['filters2'], kernel_size=3, activation='elu'),
            tf.keras.layers.MaxPooling1D(pool_size=3), # MaxPooling1D downsizes the model by 2, but the output of the previous one without orient is size 1
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.config['units1'], activation='elu'),
            tf.keras.layers.Dense(self.config['units2'], activation='elu'),
            tf.keras.layers.Dense(self.config['units3'], activation='elu'),
            tf.keras.layers.Dense(self.config['units4'], activation='elu'),
            tf.keras.layers.Dense(self.config['units5'], activation='elu'),
            tf.keras.layers.Dense(output_size, activation='elu'),
        ])
        
        self.model.summary()
        tf.keras.utils.plot_model(self.model, 'tiago_dnn_cnn/cnn_model.png', show_shapes=True)

    def save(self):
        # Save model
        if self.config['save']:
            self.model.save('cnn.keras')
            print('Trained model saved to .keras file')
