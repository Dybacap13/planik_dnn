#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import SwitchController


dual = True


if __name__ == "__main__":
  rospy.init_node("init_datagen")

  rospy.wait_for_message("/joint_states", JointState)
  rospy.sleep(3.0)

  rospy.wait_for_service("/controller_manager/switch_controller")
  manager = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
  rospy.loginfo("Switching controllers...")

  if dual:
    response = manager(start_controllers=['datagen_controller'], stop_controllers=['arm_right_controller'], strictness=2)
  else:
    response = manager(start_controllers=['datagen_controller'], stop_controllers=['arm_rightcontroller'], strictness=2)

  if not response.ok:
    rospy.logfatal("Failed to switch controllers")

  rospy.loginfo("...done.")
