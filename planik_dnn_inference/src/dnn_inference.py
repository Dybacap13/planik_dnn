#!/usr/bin/env python3

import os
import yaml
import rclpy
from rclpy.node import Node
import tensorflow as tf
import pandas as pd
from threading import Lock
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ament_index_python.packages import get_package_share_directory


class ModelInferNode(Node):
    def __init__(self):
        super().__init__('model_infer')
        
        # Get package share directory
        package_share_dir = get_package_share_directory('planik_dnn_inference')
        
        # Load configuration
        config_path = os.path.join(package_share_dir, 'config', 'tiago_dnn.yaml')
        self.get_logger().info(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            tiago_info = yaml.safe_load(f)

        # Load stats
        stats_path = os.path.join(package_share_dir, 'models', 'stats', tiago_info['stats'])
        self.get_logger().info(f"Loading stats from: {stats_path}")
        
        with open(stats_path, 'r') as f:
            self.stats = yaml.safe_load(f)
        
        # Load model
        model_path = os.path.join(package_share_dir, 'models', 'dnn', tiago_info['model'])
        self.get_logger().info(f"Loading model from: {model_path}")
        
        self.model = tf.keras.saving.load_model(model_path)

        # Create publisher and subscriber
        self.pub = self.create_publisher(JointTrajectory, '/arm_controller/command', 1)
        self.sub = self.create_subscription(Pose, '/infer/state_dnn', self.callback, 1)
        
        self.mutex = Lock()
        
        self.get_logger().info(f"Created model inference pub and sub from model {tiago_info['model']}")

    def callback(self, msg):
        self.get_logger().info("Got a target")
        input_og = [msg.position.x, msg.position.y, msg.position.z,
                   msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        input_df = pd.DataFrame(input_og)  # Renamed to avoid shadowing built-in input

        norm = self.stats['norm']

        # Normalization of input
        if norm == 'std':
            input_df = (input_df - pd.DataFrame(self.stats['df_mean_in'])) / pd.DataFrame(self.stats['df_std_in'])
        elif norm == 'norm':
            input_df = (input_df - pd.DataFrame(self.stats['df_min_in'])) / (pd.DataFrame(self.stats['df_max_in']) - pd.DataFrame(self.stats['df_min_in']))
        elif norm == 'max-abs':
            input_df = input_df / pd.DataFrame(self.stats['df_maxabs_in'])
        elif norm == 'iqr':
            input_df = (input_df - pd.DataFrame(self.stats['df_median_in'])) / (pd.DataFrame(self.stats['df_quantile75_in']) - pd.DataFrame(self.stats['df_quantile25_in']))

        input_array = input_df.to_numpy().transpose()
        
        # Inference
        self.mutex.acquire()
        try:
            output = self.model.predict(input_array)
            output_df = pd.DataFrame(output).transpose()

            # De-normalization of output
            if norm == 'std':
                output_df = output_df * pd.DataFrame(self.stats['df_std_out']) + pd.DataFrame(self.stats['df_mean_out'])
            elif norm == 'norm':
                output_df = output_df * (pd.DataFrame(self.stats['df_max_out']) - pd.DataFrame(self.stats['df_min_out'])) + pd.DataFrame(self.stats['df_min_out'])
            elif norm == 'max-abs':
                output_df = output_df * pd.DataFrame(self.stats['df_maxabs_out'])
            elif norm == 'iqr':
                output_df = output_df * (pd.DataFrame(self.stats['df_quantile75_out']) - pd.DataFrame(self.stats['df_quantile25_out'])) + pd.DataFrame(self.stats['df_median_out'])

            # Send goal
            goal = JointTrajectory()
            points = JointTrajectoryPoint()
            goal.joint_names = ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 
                               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            points.positions = output_df.transpose().values.tolist()[0]
            points.time_from_start.sec = 5
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