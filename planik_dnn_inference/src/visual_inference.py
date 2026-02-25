#!/usr/bin/env python3

import os
import yaml
import rclpy
from rclpy.node import Node
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


class ModelInferNode(Node):
    def __init__(self):
        super().__init__('model_infer')
        
        # Load configuration
        with open(os.path.join('/docker-ros/ws/src/tiago-inference/config', 'tiago_dnn.yaml'), 'r') as f:
            tiago_info = yaml.safe_load(f)

        with open(os.path.join('/docker-ros/ws/src/tiago-inference/models/stats', tiago_info['stats']), 'r') as f:
            self.stats = yaml.safe_load(f)
            
        self.model = tf.keras.saving.load_model(os.path.join('/docker-ros/ws/src/tiago-inference/models/dnn', tiago_info['model']))

        # Create publisher and subscriber
        if dual:
            self.pub = self.create_publisher(JointTrajectory, '/arm_right_controller/command', 1)
        else:
            self.pub = self.create_publisher(JointTrajectory, '/arm_controller/command', 1)
            
        self.sub = self.create_subscription(Point, '/rgbd_detection_coords', self.callback, 1)
        
        self.mutex = Lock()
        
        self.get_logger().info(f"Created model inference pub and sub from model {tiago_info['model']}")

    def callback(self, msg):
        self.get_logger().info("Got a target")
        
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

        norm = self.stats['norm']

        # Normalization of input
        if norm == 'std':
            input = (input - pd.DataFrame(self.stats['df_mean_in'])) / pd.DataFrame(self.stats['df_std_in'])
        elif norm == 'norm':
            input = (input - pd.DataFrame(self.stats['df_min_in'])) / (pd.DataFrame(self.stats['df_max_in']) - pd.DataFrame(self.stats['df_min_in']))
        elif norm == 'max-abs':
            input = input / pd.DataFrame(self.stats['df_maxabs_in'])
        elif norm == 'iqr':
            input = (input - pd.DataFrame(self.stats['df_median_in'])) / (pd.DataFrame(self.stats['df_quantile75_in']) - pd.DataFrame(self.stats['df_quantile25_in']))

        input = input.to_numpy().transpose()
        
        # Inference
        self.mutex.acquire()
        try:
            output = self.model.predict(input)
            output = pd.DataFrame(output).transpose()

            # De-normalization of output
            if norm == 'std':
                output = output * pd.DataFrame(self.stats['df_std_out']) + pd.DataFrame(self.stats['df_mean_out'])
            elif norm == 'norm':
                output = output * (pd.DataFrame(self.stats['df_max_out']) - pd.DataFrame(self.stats['df_min_out'])) + pd.DataFrame(self.stats['df_min_out'])
            elif norm == 'max-abs':
                output = output * pd.DataFrame(self.stats['df_maxabs_out'])
            elif norm == 'iqr':
                output = output * (pd.DataFrame(self.stats['df_quantile75_out']) - pd.DataFrame(self.stats['df_quantile25_out'])) + pd.DataFrame(self.stats['df_median_out'])

            # Send goal
            self.get_logger().info(f"Output: {output.transpose().values.tolist()[0]}")

            goal = JointTrajectory()
            points = JointTrajectoryPoint()
            
            if dual:
                goal.joint_names = ['arm_right_1_joint', 'arm_right_2_joint', 'arm_right_3_joint', 
                                   'arm_right_4_joint', 'arm_right_5_joint', 'arm_right_6_joint', 'arm_right_7_joint']
            else:
                goal.joint_names = ['arm_1_joint', 'arm_2_joint', 'arm_3_joint', 'arm_4_joint', 
                                   'arm_5_joint', 'arm_6_joint', 'arm_7_joint']
            
            points.positions = output.transpose().values.tolist()[0]
            points.time_from_start.sec = 5  # Changed to Duration in ROS2
            goal.points.append(points)
            self.pub.publish(goal)

        finally:
            self.mutex.release()


def main(args=None):
    rclpy.init(args=args)
    node = ModelInferNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()