#!/usr/bin/env python
# From https://www.sciencedirect.com/science/article/pii/S0952197623004852 Fig. 14. Beware, description and image do not correspond
# See also https://easychair.org/publications/paper/J1qk/open, https://ieeexplore.ieee.org/document/10217721

import yaml
import numpy as np
import tensorflow as tf


class Lstm():
    def __init__(self):
        with open('tiago_dnn_rnn/rnn_config.yaml') as f:
            self.config = yaml.safe_load(f)['lstm']

        x_train = np.load(f"data/{self.config['data_dir']}/x_train_curr1.npy")
        y_train = np.load(f"data/{self.config['data_dir']}/y_train_curr1.npy")

        self.input_size = x_train.shape[1]
        output_size = y_train.shape[1]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(1000, return_sequences=True, input_shape=(self.input_size, 1)),
            tf.keras.layers.Dense(1000, activation='swish'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(800, return_sequences=True),
            tf.keras.layers.Dense(800, activation='swish'),
            tf.keras.layers.LSTM(600, return_sequences=True),
            tf.keras.layers.Dense(600, activation='swish'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(400, return_sequences=True),
            tf.keras.layers.Dense(400, activation='swish'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation='swish'),
            tf.keras.layers.LSTM(100, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(output_size)
        ])

        self.model.summary()
        tf.keras.utils.plot_model(self.model, 'tiago_dnn_rnn/lstm_model.png', show_shapes=True)

    def save(self):
        # Save model
        if self.config['save']:
            self.model.save('lstm_rnn.keras')
            print('Trained model saved to .keras file')
