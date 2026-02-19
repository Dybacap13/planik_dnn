# TIAGO-INFERENCE

This module needs tensorflow and ros to work. The docker image used for this purpose can be found [here](https://hub.docker.com/r/rwthika/ros-ml). To build it and incorporate the necessary changes, run:

```bash
bash docker/build.sh
```

Once this is done, one can run the image from the `tiago-inference` folder by using:

```bash
bash docker/run.sh
```

In the `/docker-ros/ws/` workspace, run `catkin build` to build the package then `source devel/setup.bash` and finally

```bash
rosrun tiago_dnn_inference dnn_inference.py
```

If the controller is running in a different computer or in the real robot, remember to set **ROS_MASTER_URI** and **ROS_IP**.