#!/usr/bin/env python3.9

import os
import yaml
import rospy
import tensorflow as tf
import pandas as pd
from threading import Lock
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def callback(msg):
    rospy.loginfo("got a target")
    input_og = [msg.position.x, msg.position.y, msg.position.z,
             msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
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
    goal = JointTrajectory()
    points = JointTrajectoryPoint()
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

    pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
    rospy.Subscriber('/infer/state_dnn', Pose, callback, queue_size=1)

    rospy.loginfo(f"created model inference pub and sub from model {tiago_info['model']}")
    mutex = Lock()

    rospy.spin()
