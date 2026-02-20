#!/usr/bin/env python
# See https://github.com/dcagigas/Robotics/blob/main/Scorbot_ER_VII/inverse_kinematics_ANN/Scorbot_ANN_bootstrapping_96/scorbot_make_models_inverse_kinematics.py

import yaml
# import wandb
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform


class SixSubnetMlp():
    def __init__(self):
        with open('planik_dnn_ik/planik_dnn_mlp/mlp_config.yaml') as f:
            self.config = yaml.safe_load(f)['six_subnet_mlp']

        x_train = np.load(f"data/{self.config['data_dir']}/x_train_curr1.npy")
        y_train = np.load(f"data/{self.config['data_dir']}/y_train_curr1.npy")

        self.input_size = x_train.shape[1]

        # subnet 1:
        input = keras.Input(shape=(self.input_size,), name="input")
        layer1 = layers.Dense(self.config['units1'], activation="sigmoid", name="layer1") (input)
        output1 = layers.Dense(1, activation = "linear", name="q1")(layer1)

        # subnet 2:
        input2 = layers.concatenate([input], name="input2")
        layer2 = layers.Dense(self.config['units2'], activation="relu", name="layer2") (input2)
        output2 = layers.Dense(1, activation = "linear", name="q2")(layer2)

        # subnet 3:
        input3 = layers.concatenate([input, output2], name="input3")    
        layer3 = layers.Dense(self.config['units3'], activation="relu", name="layer3") (input3)
        output3 = layers.Dense(1, activation = "linear", name="q3")(layer3)

        # subnet 4:
        input4 = layers.concatenate([input, output3], name="input4")
        layer4 = layers.Dense(self.config['units4'], activation="relu", name="layer4") (input4)
        output4 = layers.Dense(1, activation = "linear", name="q4")(layer4)

        # subnet 5:
        input5 = layers.concatenate([input, output4], name="input5")
        layer5 = layers.Dense(self.config['units5'], activation="relu", name="layer5") (input5)
        output5 = layers.Dense(1, activation = "linear", name="q5")(layer5)

        # subnet 6:
        input6 = layers.concatenate([input, output5], name="input6")    
        layer6 = layers.Dense(self.config['units6'], activation="relu", name="layer6") (input6)
        output6 = layers.Dense(1, activation = "linear", name="q6")(layer6)

       

        # combined outputs
        combined_outputs = layers.concatenate([output1, output2, output3, output4, output5, output6], name="combined_output")

        self.model = keras.Model(inputs=[input], outputs=[combined_outputs], name="ConcatMLP")
        
        self.model.summary()
        keras.utils.plot_model(self.model, 'planik_dnn_ik/planik_dnn_mlp/concat_mlp_model.png', show_shapes=True)

    def save(self):
        # Save model
        if self.config['save']:
            self.model.save('planik_dnn_ik/planik_dnn_mlp/6subnet_mlp.keras')
            print('Trained model saved to .keras file')
