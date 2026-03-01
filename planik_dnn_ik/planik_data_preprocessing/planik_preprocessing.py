#!/usr/bin/env python

import os
import yaml
import argparse
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform

from ament_index_python.packages import get_package_share_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="choose input csv filename", type=str)
    parser.add_argument("--norm", help="choose normalization type: 0 for standardization, 1 for min-max normalization, "
                                        "2 for max-absolute, 3 por robusy scaling, other for none")
    parser.add_argument("--name", help="choose name for split data files")
    parser.add_argument("--curr", help="choose number of curriculums generated for the data", type=int)
    args = parser.parse_args()

    package_path = get_package_share_directory('planik_models')
    save_dir = os.path.join(package_path, args.name, 'preprocessing')
    csv_file = os.path.join(package_path, args.name)


    df = pd.read_csv(f'd{csv_file}.csv')
    print(f'Read {args.file}.csv. Here is a sample: ')
    print(df.sample())

    x_cols = ['ee_x', 'ee_y', 'ee_z']
    if 'ee_quat_x' in df.columns:
        x_cols = x_cols + ['ee_quat_x', 'ee_quat_y', 'ee_quat_z', 'ee_quat_w']
    elif 'ee_rot_20' in df.columns:
        x_cols = x_cols + ['ee_rot_00', 'ee_rot_01', 'ee_rot_02', 'ee_rot_10', 'ee_rot_11', 'ee_rot_12', 'ee_rot_20', 'ee_rot_21', 'ee_rot_22']
    elif 'ee_rot_00' in df.columns:
        x_cols = x_cols + ['ee_rot_00', 'ee_rot_01', 'ee_rot_02', 'ee_rot_10', 'ee_rot_11', 'ee_rot_12']

 
    y_cols = ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    # Remove duplicates
    print(f'Dropping {df.duplicated(x_cols).sum()} duplicates')
    df = df.drop_duplicates(x_cols, keep='first')

    # Get stats
    data_stats = {'df_mean_in': df[x_cols].mean().tolist(), 
                  'df_mean_out': df[y_cols].mean().tolist(),
                  'df_std_in': df[x_cols].std().tolist(), 
                  'df_std_out': df[y_cols].std().tolist(),
                  'df_median_in': df[x_cols].median().tolist(), 
                  'df_median_out': df[y_cols].median().tolist(),
                  'df_max_in': df[x_cols].max().tolist(), 
                  'df_max_out': df[y_cols].max().tolist(),
                  'df_maxabs_in': df[x_cols].abs().max().tolist(), 
                  'df_maxabs_out': df[y_cols].abs().max().tolist(),
                  'df_min_in': df[x_cols].min().tolist(), 
                  'df_min_out': df[y_cols].min().tolist(),
                  'df_quantile25_in': df[x_cols].quantile(0.25).tolist(),
                  'df_quantile25_out': df[y_cols].quantile(0.25).tolist(),
                  'df_quantile75_in': df[x_cols].quantile(0.75).tolist(),
                  'df_quantile75_out': df[y_cols].quantile(0.75).tolist()}

    # Standardization (Z-scaling)
    if args.norm == '0':
        df = (df - df.mean()) / df.std()
        data_stats['norm'] = 'std'

    # Normalization (min-max normalization)
    elif args.norm == '1':
        df = (df - df.min()) / (df.max() - df.min())
        data_stats['norm'] = 'norm'

    # max-absolute scaling
    elif args.norm == '2':
        df = df / df.abs().max()
        data_stats['norm'] = 'max-abs'

    # IQR scaling
    elif args.norm == '3':
        df = (df - df.median()) / (df.quantile(0.75) - df.quantile(0.25))
        data_stats['norm'] = 'iqr'

    # None
    else:
        data_stats['norm'] = 'none'

    # Generate curriculums
    file_path = 'urdf/tiago/tiago.urdf'
    robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
    thetas = [0, 0, 0, 0, 0, 0, 0]
    shoulder = robot.forward_kin(thetas)['arm_1_link'].pos

    

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Validation and test set
    x = df[x_cols].to_numpy()
    y = df[y_cols].to_numpy()
    val_size = 0.1
    test_size = 0.2
    x, x_val, y, y_val = train_test_split(x, y, test_size=val_size)
    x, x_test, y, y_test = train_test_split(x, y, test_size=test_size)
    df = pd.DataFrame(np.concatenate((x, y), axis=1), columns=x_cols + y_cols)

    with open(f'{save_dir}/x_val.npy', 'wb') as f:
        np.save(f, x_val)

    with open(f'{save_dir}/y_val.npy', 'wb') as f:
        np.save(f, y_val)

    with open(f'{save_dir}/x_test.npy', 'wb') as f:
        np.save(f, x_test)

    with open(f'{save_dir}/y_test.npy', 'wb') as f:
        np.save(f, y_test)


    # Equal distance 
    df['distance'] = np.sqrt(np.square(df['ee_x'] - shoulder[0]) + np.square(df['ee_y'] - shoulder[1]) + np.square(df['ee_z'] - shoulder[2]))
    bounds = np.linspace(min(df['distance']), max(df['distance']), args.curr + 1)

    # Equal number of samples
    # bounds = df['distance'].quantile(np.linspace(0, 1, args.curr + 1)).to_list()

    curr_sizes = []
    for i in range(args.curr):
        df_curr = df[df['distance'].between(bounds[0], bounds[i+1])]
        curr_sizes.append(df_curr.shape[0])

        x_train = df_curr[x_cols].to_numpy()
        y_train = df_curr[y_cols].to_numpy()

        # Save
        with open(f'{save_dir}/x_train_curr{i+1}.npy', 'wb') as f:
            np.save(f, x_train)

        with open(f'{save_dir}/y_train_curr{i+1}.npy', 'wb') as f:
            np.save(f, y_train)

    with open(f'{save_dir}/data_stats.yaml', 'w') as f:
        data_stats['test_size'] = test_size
        data_stats['curriculums'] = args.curr
        data_stats['curriculum_sizes'] = curr_sizes
        yaml.dump(data_stats, f)

    print(f'Done! Files split and saved {save_dir}')
    print(f'{args.curr} curriculums have been generated, with sizes {curr_sizes}')