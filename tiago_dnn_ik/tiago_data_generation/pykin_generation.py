#!/usr/bin/env python

import argparse
import random
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pykin.robots.single_arm import SingleArm
from pykin.kinematics.transform import Transform
from pykin.utils import plot_utils as p_utils


def show_info(robot):
    robot.show_robot_info()

    thetas = [0, 0, 0, 0, 0, 0, 0]
    robot.set_transform(thetas)

    _, ax = p_utils.init_3d_figure("FK", visible_axis=True)
    p_utils.plot_robot(ax=ax, 
                robot=robot,
                geom="visual",
                only_visible_geom=True,
                alpha=1)
    p_utils.show_figure()


def show_stats(file, orient):
    df = pd.read_csv(f'{file}.csv')
    df['distance'] = np.sqrt(np.square(df['ee_x']) + np.square(df['ee_y']) + np.square(df['ee_z']))

    vbles = ['ee_x', 'ee_y', 'ee_z', 'distance']

    # Generated data
    if orient == 'quaternion':
        print(df[vbles + ['ee_quat_x','ee_quat_y','ee_quat_z','ee_quat_w']].describe())
    elif orient == 'matrix':
        print(df[vbles + ['ee_rot_00','ee_rot_01','ee_rot_02','ee_rot_10','ee_rot_11','ee_rot_12','ee_rot_20','ee_rot_21','ee_rot_22']].describe())
    else:
        print(df[vbles].describe())

    plt.figure(figsize=(20, 20))
    colors = plt.cm.viridis(np.linspace(0, 1, len(vbles)))

    for i, col in enumerate(vbles):
        plt.subplot(2, 3, i + 1)
        plt.hist(df[col], color=colors[i])
        plt.title(col)
        plt.tight_layout()

    plt.savefig(f'{file}_stats.png')
    print(f'See data stats in {file}_stats.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rob", help="choose robot: tiago or dual", type=str)
    parser.add_argument("--num", help="choose number of samples to generate", type=int)
    parser.add_argument("--file", help="choose resulting csv filename", type=str)
    parser.add_argument("--orient", help="choose orientation (quaternion, matrix or none)")
    args = parser.parse_args()

    if args.rob == 'tiago':
        file_path = 'urdf/tiago/tiago.urdf'
    elif args.rob == 'dual':
        file_path = 'urdf/tiago/tiago_dual.urdf'
    robot = SingleArm(file_path, Transform(rot=[0.0, 0.0, 0.0], pos=[0, 0, 0]))
    joint_names = robot.get_all_active_joint_names()
    # show_info(robot)

    limits = {}
    dist_to_limits = 0.2
    for joint in joint_names:
        limits[joint] = {'lower': robot.joints[joint].limit[0] + dist_to_limits, 'upper': robot.joints[joint].limit[1] - dist_to_limits}

    # generate positions
    i = 0
    with open(f'data/pykin/{args.file}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        if args.orient == 'quaternion':
            csvwriter.writerow(['arm_1','arm_2','arm_3','arm_4','arm_5','arm_6','arm_7','ee_x','ee_y','ee_z',
                               'ee_quat_x','ee_quat_y','ee_quat_z','ee_quat_w'])
        elif args.orient == 'matrix':
            csvwriter.writerow(['arm_1','arm_2','arm_3','arm_4','arm_5','arm_6','arm_7','ee_x','ee_y','ee_z',
                               'ee_rot_00','ee_rot_01','ee_rot_02','ee_rot_10','ee_rot_11','ee_rot_12','ee_rot_20','ee_rot_21','ee_rot_22'])
        elif args.orient == '6d':
            csvwriter.writerow(['arm_1','arm_2','arm_3','arm_4','arm_5','arm_6','arm_7','ee_x','ee_y','ee_z',
                               'ee_rot_00','ee_rot_01','ee_rot_02','ee_rot_10','ee_rot_11','ee_rot_12'])
        else:
            csvwriter.writerow(['arm_1','arm_2','arm_3','arm_4','arm_5','arm_6','arm_7','ee_x','ee_y','ee_z'])

        pbar = tqdm(total=args.num)
        while i < args.num:
            pos = []
            for joint in joint_names:
                pos.append(random.uniform(limits[joint]['lower'], limits[joint]['upper']))

            fk = robot.forward_kin(pos)
            if args.rob == 'tiago':
                shoulder_pos = fk['arm_1_link']
                fk = fk['tiago_link_ee']
            elif args.rob == 'dual':
                shoulder_pos = fk['arm_right_1_link']
                fk = fk['arm_right_7_link']

            if fk.pos[2] > 0.3 and np.linalg.norm(shoulder_pos.pos - fk.pos) < 0.8:
                if args.orient == 'quaternion':
                    csvwriter.writerow(pos + fk.pos.tolist() + fk.rot.tolist())
                elif args.orient == 'matrix':
                    csvwriter.writerow(pos + fk.pos.tolist() + fk.rotation_matrix.flatten().tolist())
                elif args.orient == '6d':
                    csvwriter.writerow(pos + fk.pos.tolist() + fk.rotation_matrix.flatten().tolist()[:-3])
                else:
                    csvwriter.writerow(pos + fk.pos.tolist())
                
                i += 1
                pbar.update(1)

        pbar.close()

    print(f'Data saved to data/pykin/{args.file}.csv')
    show_stats(f'data/pykin/{args.file}', args.orient)
