#!/usr/bin/env python

import os
import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", help="choose normalization type: 0 for standardization, 1 for min-max normalization, other for none")
    parser.add_argument("--name", help="choose name for split data files")
    args = parser.parse_args()

    with open(os.path.join('/tiago_public_ws/src/tiago_data_generation/', 'config/datagen_controller.yaml')) as f:
        config = yaml.safe_load(f)

    print('Reading csv dataset from ' + config['datagen_controller']['filename'])
    df = pd.read_csv(os.path.join('/home/anacg/.ros', config['datagen_controller']['filename']), lineterminator='\n')
    print(df.sample())

    x_cols = ['ee_x', 'ee_y', 'ee_z']
    if config['datagen_controller']['orient'] == 'quaternion':
        x_cols = x_cols + ['ee_quat_x', 'ee_quat_y', 'ee_quat_z', 'ee_quat_w']
    elif config['datagen_controller']['orient'] == 'matrix':
        x_cols = x_cols + ['ee_rot_00', 'ee_rot_01', 'ee_rot_02', 'ee_rot_10', 'ee_rot_11', 'ee_rot_12', 'ee_rot_20', 'ee_rot_21', 'ee_rot_22']

    y_cols = ['arm_1', 'arm_2', 'arm_3', 'arm_4', 'arm_5', 'arm_6', 'arm_7']

    data_stats = {'df_mean_in': df[x_cols].mean().tolist(), 
                  'df_mean_out': df[y_cols].mean().tolist(),
                  'df_std_in': df[x_cols].std().tolist(), 
                  'df_std_out': df[y_cols].std().tolist(),
                  'df_max_in': df[x_cols].max().tolist(), 
                  'df_max_out': df[y_cols].max().tolist(),
                  'df_min_in': df[x_cols].min().tolist(), 
                  'df_min_out': df[y_cols].min().tolist()}

    # Standardization
    if args.norm == '0':
        df = (df - df.mean()) / df.std()
        data_stats['norm'] = 'std'

    # Normalization
    elif args.norm == '1':
        df = (df - df.min()) / (df.max() - df.min())
        data_stats['norm'] = 'norm'

    # None
    else:
        data_stats['norm'] = 'none'

    x = df[x_cols].to_numpy()
    y = df[y_cols].to_numpy()

    # Split and save
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/gazebo/' + args.name + '/x_train_curr1.npy'), 'w') as f:
        np.save(f, x_train)

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/gazebo/' + args.name + '/y_train_curr1.npy'), 'w') as f:
        np.save(f, y_train)

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/gazebo/' + args.name + '/x_test_curr1.npy'), 'w') as f:
        np.save(f, x_test)

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/gazebo/' + args.name + '/y_test_curr1.npy'), 'w') as f:
        np.save(f, y_test)

    with open(os.path.join('/home/anacg/repos/tiago/tiago_dnn_ik/data/gazebo/' + args.name + 'data_stats.yaml'), 'w') as f:
        yaml.dump(data_stats, f)
