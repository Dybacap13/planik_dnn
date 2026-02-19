#!/usr/bin/env python

import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt


class CnnModel(kt.HyperModel):
    def build(self, hp):
        # params
        hp_filters_1 = hp.Int('filters1', min_value=100, max_value=1000, step=200)
        hp_filters_2 = hp.Int('filters2', min_value=100, max_value=1000, step=200)
        hp_units_1 = hp.Int('units1', min_value=100, max_value=800, step=100)
        hp_units_2 = hp.Int('units2', min_value=100, max_value=800, step=100)
        hp_units_3 = hp.Int('units3', min_value=100, max_value=800, step=100)
        hp_units_4 = hp.Int('units4', min_value=100, max_value=800, step=100)
        hp_units_5 = hp.Int('units5', min_value=100, max_value=800, step=100)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # Specify the model's architecture
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=hp_filters_1, kernel_size=1, input_shape=(input_size, 1)),
            tf.keras.layers.Conv1D(filters=hp_filters_2, kernel_size=3, activation='elu'),
            # tf.keras.layers.MaxPooling1D(pool_size=3), # MaxPooling1D downsizes the model by 2, but the output of the previous one without orient is size 1
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(hp_units_1, activation='elu'),
            tf.keras.layers.Dense(hp_units_2, activation='elu'),
            tf.keras.layers.Dense(hp_units_3, activation='elu'),
            tf.keras.layers.Dense(hp_units_4, activation='elu'),
            tf.keras.layers.Dense(hp_units_5, activation='elu'),
            tf.keras.layers.Dense(output_size, activation='elu'),
        ])

        # Specify the loss fuction, optimizer, metrics
        model.compile(
            loss = 'mean_squared_error',
            optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate),
            metrics = ['accuracy']
        )

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [16, 32]),
            **kwargs,
        )


if __name__ == '__main__':
    with open('cnn_config.yaml') as f:
        config = yaml.safe_load(f)

    x_train = np.load(f"../data/{config['x_train_file']}")
    y_train = np.load(f"../data/{config['y_train_file']}")
    x_test = np.load(f"../data/{config['x_test_file']}")
    y_test = np.load(f"../data/{config['y_test_file']}")

    print("x_train shape :", x_train.shape)
    print("x_test shape :", x_test.shape)
    print("y_train shape :", y_train.shape)
    print("y_test shape :", y_test.shape)

    input_size = x_train.shape[1]
    output_size = y_train.shape[1]

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    print('Tuning hyperparameters for cnn model')
    tuner = kt.Hyperband(CnnModel(),
                        objective='val_accuracy',
                        max_epochs=100,
                        factor=3,
                        directory='cnn_tuning',
                        project_name='cnn')
    tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'Best hyperparameters are: {best_hps}')

    # save hyperparams
    config['lr'] = best_hps.get('learning_rate')
    config['units'] = best_hps.get('units')
    config['batch_size'] = best_hps.get('batch_size')

    with open('cnn_config.yaml', 'r') as f:
        yaml.dump(config, f)
