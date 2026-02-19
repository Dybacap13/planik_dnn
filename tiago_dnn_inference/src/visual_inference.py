#!/usr/bin/env python3.9

import os
import yaml
import rospy
import tensorflow as tf
import pandas as pd
import numpy as np
from threading import Lock
from geometry_msgs.msg import Point
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


dual = True


def rotationMatrixToQuaternion1(m):
    #q0 = qw
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if(t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2,1] - m[1,2]) * t
        q[1] = (m[0,2] - m[2,0]) * t
        q[2] = (m[1,0] - m[0,1]) * t

    else:
        i = 0
        if (m[1,1] > m[0,0]):
            i = 1
        if (m[2,2] > m[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3

        t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k,j] - m[j,k]) * t
        q[j] = (m[j,i] + m[i,j]) * t
        q[k] = (m[k,i] + m[i,k]) * t

    return q


def callback(msg):
    rospy.loginfo("got a target")
    
    H_root_cam = np.array([[1.0000000, 0.0000180, -0.0002330, 0.124999],
                           [-0.0002330, -0.0000047, -1.0000000, 0.00],
                           [-0.0000180,  1.0000000, -0.0000047, 1.126957],
                           [0, 0, 0, 1]])
    
    H_cam_obj = np.array([[1, 0, 0, msg.x],
                          [0, 1, 0, msg.y],
                          [0, 0, 1, msg.z],
                          [0, 0, 0, 1]])
    
    H_root_cam = np.matmul(H_root_cam, H_cam_obj)

    rot_m = np.array([[H_root_cam[0, 0], H_root_cam[0, 1], H_root_cam[0, 2]], 
                      [H_root_cam[1, 0], H_root_cam[1, 1], H_root_cam[1, 2]], 
                      [H_root_cam[2, 0], H_root_cam[2, 1], H_root_cam[2, 2]]])
    rot_q = rotationMatrixToQuaternion1(rot_m)
    pos = [H_root_cam[0, 3], H_root_cam[1, 3], H_root_cam[2, 3]]
    
    input_og = pos + rot_q.tolist()
    input = pd.DataFrame(input_og)

    norm = stats['norm']

    # normalization of input
    if norm == 'std':
        input = (input - pd.DataFrame(stats['df_mean_in'])) / pd.DataFrame(stats['df_std_in'])
    elif norm == 'norm':
        input = (input - pd.DataFrame(stats['df_min_in'])) / (pd.DataFrame(stats['df_max_in']) - pd.DataFrame(stats['df_min_in']))
    elif norm == 'max-abs':
        input = input / pd.DataFrame(stats['df_maxabs_in'])
    elif norm == 'iqr':
        input = (input - pd.DataFrame(stats['df_median_in'])) / (pd.DataFrame(stats['df_quantile75_in']) - pd.DataFrame(stats['df_quantile25_in']))

    input = input.to_numpy().transpose()
    
    # inference
    mutex.acquire()
    output = model.predict(input)
    output = pd.DataFrame(output).transpose()

    # de-normalizaion of output
    if norm == 'std':
        output = output * pd.DataFrame(stats['df_std_out']) + pd.DataFrame(stats['df_mean_out'])
    elif norm == 'norm':
        output = output * (pd.DataFrame(stats['df_max_out']) - pd.DataFrame(stats['df_min_out'])) + pd.DataFrame(stats['df_min_out'])
    elif norm == 'max-abs':
        output = output * pd.DataFrame(stats['df_maxabs_out'])
    elif norm == 'iqr':
        output = output * (pd.DataFrame(stats['df_quantile75_out']) - pd.DataFrame(stats['df_quantile25_out'])) + pd.DataFrame(stats['df_median_out'])

    # Send goal
    rospy.loginfo(output.transpose().values.tolist()[0])

    goal = JointTrajectory()
    points = JointTrajectoryPoint()
    if dual:
        goal.joint_names = ['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint', 'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint']
    else:
        goal.joint_names = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
    points.positions = output.transpose().values.tolist()[0]
    points.time_from_start = rospy.Duration(5)
    goal.points.append(points)
    pub.publish(goal)

    mutex.release()


if __name__ == '__main__':
    with open(os.path.join('/docker-ros/ws/src/tiago-inference/config', 'tiago_dnn.yaml'), 'r') as f:
        tiago_info = yaml.safe_load(f)

    with open(os.path.join('/docker-ros/ws/src/tiago-inference/models/stats', tiago_info['stats']), 'r') as f:
        stats = yaml.safe_load(f)
        
    model = tf.keras.saving.load_model(os.path.join('/docker-ros/ws/src/tiago-inference/models/dnn', tiago_info['model']))

    rospy.init_node('model_infer')
    rate = rospy.Rate(10)   # 10hz

    if dual:
        pub = rospy.Publisher('/arm_right_controller/command', JointTrajectory, queue_size=1)
    else:
        pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
    rospy.Subscriber('/rgbd_detection_coords', Point, callback, queue_size=1)

    rospy.loginfo(f"created model inference pub and sub from model {tiago_info['model']}")
    mutex = Lock()

    rospy.spin()
