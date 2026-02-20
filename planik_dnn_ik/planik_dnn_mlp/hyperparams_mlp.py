#!/usr/bin/env python

import yaml
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt


class SimpleMlpModel(kt.HyperModel):
    def build(self, hp):
        # params
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        # Specify the model's architecture
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape = [input_size]),
            tf.keras.layers.Dense(units=hp_units, activation='relu'),
            tf.keras.layers.Dense(output_size, activation='softmax'),
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
            batch_size=hp.Choice("batch_size", [8, 16, 32, 64]),
            **kwargs,
        )
    

class SevenSubnetMlpModel(kt.HyperModel):
    def build(self, hp):
        # params
        hp_units_1 = hp.Int('units1', min_value=32, max_value=512, step=32)
        hp_units_2 = hp.Int('units2', min_value=32, max_value=512, step=32)
        hp_units_3 = hp.Int('units3', min_value=32, max_value=512, step=32)
        hp_units_4 = hp.Int('units4', min_value=32, max_value=512, step=32)
        hp_units_5 = hp.Int('units5', min_value=32, max_value=512, step=32)
        hp_units_6 = hp.Int('units6', min_value=32, max_value=512, step=32)
        hp_units_7 = hp.Int('units7', min_value=32, max_value=512, step=32)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        inputX = keras.Input(shape=(1,), name="Xcoor")
        inputY = keras.Input(shape=(1,), name="Ycoor")
        inputZ = keras.Input(shape=(1,), name="Zcoor")

        # subnet 1:
        base_input = layers.concatenate([inputX, inputY, inputZ], name="base_input")
        layer1 = layers.Dense(units=hp_units_1, activation="sigmoid", name="layer1")(base_input)
        output1 = layers.Dense(1, activation = "linear", name="q1")(layer1)

        # subnet 2:
        input2 = layers.concatenate([inputX, inputY, inputZ], name="input2")
        layer2 = layers.Dense(units=hp_units_2, activation="relu", name="layer2") (input2)
        output2 = layers.Dense(1, activation = "linear", name="q2")(layer2)

        # subnet 3:
        input3 = layers.concatenate([inputX, inputY, inputZ, output2], name="input3")    
        layer3 = layers.Dense(units=hp_units_3, activation="relu", name="layer3") (input3)
        output3 = layers.Dense(1, activation = "linear", name="q3")(layer3)

        # subnet 4:
        input4 = layers.concatenate([inputX, inputY, inputZ, output3], name="input4")
        layer4 = layers.Dense(units=hp_units_4, activation="relu", name="layer4") (input4)
        output4 = layers.Dense(1, activation = "linear", name="q4")(layer4)

        # subnet 5:
        input5 = layers.concatenate([inputX, inputY, inputZ, output4], name="input5")
        layer5 = layers.Dense(units=hp_units_5, activation="relu", name="layer5") (input5)
        output5 = layers.Dense(1, activation = "linear", name="q5")(layer5)

        # subnet 6:
        input6 = layers.concatenate([inputX, inputY, inputZ, output5], name="input6")    
        layer6 = layers.Dense(units=hp_units_6, activation="relu", name="layer6") (input6)
        output6 = layers.Dense(1, activation = "linear", name="q6")(layer6)

        # subnet 7:
        input7 = layers.concatenate([inputX, inputY, inputZ, output6], name="input7")
        layer7 = layers.Dense(units=hp_units_7, activation="relu", name="layer7") (input7)
        output7 = layers.Dense(1, activation = "linear", name="q7")(layer7)

        model = keras.Model(inputs=[inputX, inputY, inputZ], outputs=[output1, output2, output3, output4, output5, output6, output7], name="ConcatMLP")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("arch", help="choose mlp architecture: 0 for simple and 1 for seven_subnet")
    args = parser.parse_args()

    with open('mlp_config.yaml') as f:
        config = yaml.safe_load(f)

    x_train = np.load(f"../data/{config['x_train_file']}")
    y_train = np.load(f"../data/{config['y_train_file']}")
    x_test = np.load(f"../data/{config['x_test_file']}")
    y_test = np.load(f"../data/{config['y_test_file']}")

    x = x_train[:,[0]]
    y = x_train[:,[1]]
    z = x_train[:,[2]]

    q1_ = y_train[:,[0]]
    q2_ = y_train[:,[1]]
    q3_ = y_train[:,[2]]
    q4_ = y_train[:,[3]]
    q5_ = y_train[:,[4]]
    q6_ = y_train[:,[5]]
    q7_ = y_train[:,[6]]

    x_t = x_test[:,[0]]
    y_t = x_test[:,[1]]
    z_t = x_test[:,[2]]

    q1_t = y_test[:,[0]]
    q2_t = y_test[:,[1]]
    q3_t = y_test[:,[2]]
    q4_t = y_test[:,[3]]
    q5_t = y_test[:,[4]]
    q6_t = y_test[:,[5]]
    q7_t = y_test[:,[6]]

    print("x_train shape :", x_train.shape)
    print("x_test shape :", x_test.shape)
    print("y_train shape :", y_train.shape)
    print("y_test shape :", y_test.shape)

    input_size = x_train.shape[1]
    output_size = y_train.shape[1]

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    if args.arch == '0':
        print('Tuning hyperparameters for simple_mlp_model')
        tuner = kt.Hyperband(SimpleMlpModel(),
                            objective='val_accuracy',
                            max_epochs=100,
                            factor=3,
                            directory='mlp_tuning',
                            project_name='simple_mlp')
        tuner.search(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[stop_early])

    elif args.arch == '1':
        print('Tuning hyperparameters for seven_subnet_mlp_model')
        tuner = kt.Hyperband(SevenSubnetMlpModel(),
                            objective='val_accuracy',
                            max_epochs=100,
                            factor=3,
                            directory='mlp_tuning',
                            project_name='seven_subnet_mlp')
        tuner.search({'Xcoor':x, 'Ycoor':y, 'Zcoor':z}, {'q1': q1_, 'q2':q2_, 'q3':q3_, 'q4':q4_, 'q5':q5_, 'q6':q6_, 'q7':q7_}, 
                     epochs=50, validation_split=0.2, callbacks=[stop_early])

    else:
        print('Not an accepted option. Choose MLP architecture: 0 for simple and 1 for seven_subnet')

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'Best hyperparameters are: {best_hps}')

    # save hyperparams
    if args.arch == '0':
        config['simple_mlp']['lr'] = best_hps.get('learning_rate')
        config['simple_mlp']['units'] = best_hps.get('units')
        config['simple_mlp']['batch_size'] = best_hps.get('batch_size')
    elif args.arch == '1':
        config['seven_subnet_mlp']['lr'] = best_hps.get('learning_rate')
        config['seven_subnet_mlp']['units1'] = best_hps.get('units1')
        config['seven_subnet_mlp']['units2'] = best_hps.get('units2')
        config['seven_subnet_mlp']['units3'] = best_hps.get('units3')
        config['seven_subnet_mlp']['units4'] = best_hps.get('units4')
        config['seven_subnet_mlp']['units5'] = best_hps.get('units5')
        config['seven_subnet_mlp']['units6'] = best_hps.get('units6')
        config['seven_subnet_mlp']['units7'] = best_hps.get('units7')
        config['seven_subnet_mlp']['batch_size'] = best_hps.get('batch_size')

    with open('mlp_config.yaml', 'r') as f:
        yaml.dump(config, f)
