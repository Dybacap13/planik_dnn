#!/usr/bin/env python

import os
import sys
import yaml
import wandb
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

import custom_metrics
from tiago_dnn_mlp.simple_mlp_train import SimpleMlp
from tiago_dnn_mlp.deep_mlp_train import DeepMlp
from tiago_dnn_mlp.subnet_mlp_train import SevenSubnetMlp
from tiago_dnn_cnn.cnn_train import Cnn
from tiago_dnn_rnn.simple_rnn_train import SimpleRnn
from tiago_dnn_rnn.lstm_train import Lstm


current_curr = 1

class UpdateLRCurriculumOnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, monitor="val_loss", factor=0.1, patience=10, verbose=0, mode="auto", 
                 min_delta=1e-4, cooldown=0, min_lr=0.0, max_lr=0.1, total_currs=1, **kwargs):
        super().__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(f"ReduceLROnPlateau does not support a factor >= 1.0. Received factor={factor}")

        self.total_currs = total_currs
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter."""
        if self.mode not in {"auto", "min", "max"}:
            print(f"Learning rate reduction mode {self.mode} is unknown, fallback to auto mode.", stacklevel=2,)
            self.mode = "auto"

        if self.mode == "min" or (
            self.mode == "auto" and "acc" not in self.monitor
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.inf

        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.inf

        self.cooldown_counter = self.cooldown
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        global current_curr

        logs = logs or {}
        logs["learning_rate"] = float(np.array(self.model.optimizer.learning_rate))
        current = logs.get(self.monitor)

        if current is None:
            print(
                "Learning rate reduction is conditioned on metric "
                f"`{self.monitor}` which is not available. Available metrics "
                f"are: {','.join(list(logs.keys()))}.",
                stacklevel=2,
            )
        else:
            # Update LR
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(np.array(self.model.optimizer.learning_rate))
                    
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model.optimizer.learning_rate = new_lr

                        if self.verbose > 0:
                            print(f"\nEpoch {epoch + 1}:  Reducing learning rate to {new_lr}.")

                        self.cooldown_counter = self.cooldown
                        self.wait = 0

                    else:
                        if current_curr < self.total_currs:
                            current_curr += 1
                            print(f'\n\n--- Change to curriculum {current_curr} ---\n')
                            self.model.optimizer.learning_rate = self.max_lr
                            self._reset()
                        elif self.model.optimizer.learning_rate == self.min_lr:
                            print('All curriculums and LRs tested. Stopping...')
                            self.model.stop_training = True

    def in_cooldown(self):
        return self.cooldown_counter > 0


def train_generator(curriculums, batch_size):
    global current_curr

    while True:
        x_train = np.load(f"data/{dnn.config['data_dir']}/x_train_curr{current_curr}.npy")
        y_train = np.load(f"data/{dnn.config['data_dir']}/y_train_curr{current_curr}.npy")

        batch_x = np.zeros((batch_size, x_train.shape[1]))
        batch_y = np.zeros((batch_size, y_train.shape[1]))

        for i in range(batch_size):
            # choose random index in features
            index = random.randint(0, x_train.shape[0]-1)
            batch_x[i] = x_train[index]
            batch_y[i] = y_train[index]

        yield batch_x, batch_y


def val_generator(batch_size):
    global current_curr

    while True:
        x_test = np.load(f"data/{dnn.config['data_dir']}/x_test.npy")
        y_test = np.load(f"data/{dnn.config['data_dir']}/y_test.npy")

        batch_x = np.zeros((batch_size, x_test.shape[1]))
        batch_y = np.zeros((batch_size, y_test.shape[1]))

        for i in range(batch_size):
            # choose random index in features
            index = random.randint(0, x_test.shape[0]-1)
            batch_x[i] = x_test[index]
            batch_y[i] = y_test[index]

        yield batch_x, batch_y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="choose name of the DNN model to train", type=str)
    args = parser.parse_args()

    if args.name == 'SimpleMlp':
        dnn = SimpleMlp()
    elif args.name == 'DeepMlp':
        dnn = DeepMlp()
    elif args.name == 'SubnetMlp':
        dnn = SevenSubnetMlp()
    elif args.name == 'Cnn':
        dnn = Cnn()
    elif args.name == 'SimpleRnn':
        dnn = SimpleRnn()
    elif args.name == 'Lstm':
        dnn = Lstm()
    else:
        raise Exception('model name not recognized')
    
    with open(f"data/{dnn.config['data_dir']}/data_stats.yaml") as f:
        stats = yaml.safe_load(f)

    wandb.init(project='tiago_dnn_ik_results', name=args.name, tensorboard=True, config=dnn.config)

    # Specify the loss fuction, optimizer, metrics
    if args.name == 'SubnetMlp':
        optim = tf.keras.optimizers.RMSprop(learning_rate=dnn.config['lr'])
    else:
        optim = tf.keras.optimizers.Adam(learning_rate=dnn.config['lr'])

    dnn.model.compile(
        loss = 'mean_squared_error',
        optimizer = optim,
        metrics = ['mean_squared_error'],
        run_eagerly=True # to access individual elements in loss funct 
    )

    # Train the model
    callbacks_list = [
        keras.callbacks.TensorBoard(log_dir="logs/"),
        UpdateLRCurriculumOnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=dnn.config['verbose'], 
                                    min_lr=0.0001, max_lr=dnn.config['lr'], total_currs=stats['curriculums'], 
                                    cooldown=5),
        custom_metrics.PositionError(validation_data=val_generator(dnn.config['batch_size']), stats=stats),
        custom_metrics.OrientationError(validation_data=val_generator(dnn.config['batch_size']), stats=stats)
    ]

    # if dnn.input_size == 7:
    #     callbacks_list.append([custom_metrics.QuaternionError1(validation_data=val_generator(dnn.config['batch_size']), stats=stats),
    #                             custom_metrics.QuaternionError2(validation_data=val_generator(dnn.config['batch_size'])),
    #                             custom_metrics.QuaternionError3(validation_data=val_generator(dnn.config['batch_size']))])
        
    # elif dnn.input_size == 12:
    #     callbacks_list.append([custom_metrics.RotMatrixError1(validation_data=val_generator(dnn.config['batch_size']), stats=stats),
    #                             custom_metrics.RotMatrixError2(validation_data=val_generator(dnn.config['batch_size'])),
    #                             custom_metrics.RotMatrixError3(validation_data=val_generator(dnn.config['batch_size'])),
    #                             custom_metrics.RotMatrixError4(validation_data=val_generator(dnn.config['batch_size']))])

    history = dnn.model.fit(
        train_generator(stats['curriculums'], dnn.config['batch_size']),
        steps_per_epoch=((stats['curriculum_sizes'][0] * (1-stats['test_size'])) // dnn.config['batch_size']),
        epochs=dnn.config['epochs'],
        verbose=dnn.config['verbose'],
        callbacks=callbacks_list,
        validation_data=val_generator(dnn.config['batch_size']),
        validation_steps=((stats['curriculum_sizes'][0] * stats['test_size']) // dnn.config['batch_size'])
    )

    dnn.save()

    # Evaluate
    x_val = np.load(f"data/{dnn.config['data_dir']}/x_val.npy")
    y_val = np.load(f"data/{dnn.config['data_dir']}/y_val.npy")
    dnn.model.compile(
        loss = 'mean_squared_error',
        optimizer = optim,
        metrics = ['mean_squared_error', custom_metrics.position_error(stats), custom_metrics.orientation_error(stats)],
        run_eagerly=True # to access individual elements in loss funct 
    )

    eval = dnn.model.evaluate(x_val, y_val, batch_size=dnn.config['batch_size'], callbacks=callbacks_list)
    wandb.log({'evaluation': {
                'mse': eval[0], 
                'pos_error': eval[2], 
                'orient_error': eval[3]
                }})
