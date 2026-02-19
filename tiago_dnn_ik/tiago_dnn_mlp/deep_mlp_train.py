#!/usr/bin/env python

import yaml
import numpy as np
import tensorflow as tf


class DeepMlp():
    def __init__(self):
        with open('tiago_dnn_mlp/mlp_config.yaml') as f:
            self.config = yaml.safe_load(f)['deep_mlp']

        x_train = np.load(f"data/{self.config['data_dir']}/x_train_curr1.npy")
        y_train = np.load(f"data/{self.config['data_dir']}/y_train_curr1.npy")

        self.input_size = x_train.shape[1]
        output_size = y_train.shape[1]

        # Specify the model's architecture
        self.model = tf.keras.models.Sequential()
        for i in range(len(self.config['num_layers'])):
            num_layers = self.config['num_layers'][i]
            for j in range(num_layers):
                self.model.add(tf.keras.layers.Dense(self.config['units'][i], input_shape=[self.input_size], activation='relu'))
        self.model.add(tf.keras.layers.Dense(output_size, activation='tanh'))
        self.model.build(input_shape=[self.input_size])

        self.model.summary()
        tf.keras.utils.plot_model(self.model, 'tiago_dnn_mlp/deep_mlp_model.png', show_shapes=True)

    def save(self):
        # Save model
        if self.config['save']:
            self.model.save('tiago_dnn_mlp/deep_mlp.keras')
            print('Trained model saved to .keras file')
