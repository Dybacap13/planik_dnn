#!/usr/bin/env python
# See https://ieeexplore.ieee.org/document/10185778

import yaml
import numpy as np
import tensorflow as tf


class SimpleRnn():
    def __init__(self): 
        with open('tiago_dnn_rnn/rnn_config.yaml') as f:
            self.config = yaml.safe_load(f)['simple_rnn']

        x_train = np.load(f"data/{self.config['data_dir']}/x_train_curr1.npy")
        y_train = np.load(f"data/{self.config['data_dir']}/y_train_curr1.npy")

        # Specify the model's architecture
        self.input_size = x_train.shape[1]
        output_size = y_train.shape[1]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(self.config['units'], input_shape=(self.input_size, 1)),
            tf.keras.layers.Dense(output_size, activation='tanh'),
        ])

        self.model.summary()
        tf.keras.utils.plot_model(self.model, 'tiago_dnn_rnn/simple_rnn_model.png', show_shapes=True)

    def save(self):
        # Save model
        if self.config['save']:
            self.model.save('tiago_dnn_rnn/simple_rnn.keras')
            print('Trained model saved to .keras file')
