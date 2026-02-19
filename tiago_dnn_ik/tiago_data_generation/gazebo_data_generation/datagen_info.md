# tiago_data_generation

TiAGO data generation for learning IK. Creates a CSV file with joint angles and end-effector poses.

## Launch simulation and controller

To launch tiago_data_generation AND simulation, run:
```bash
source ~/.virtualenvs/rocker/bin/activate
```

From within `tiago-2021-accessrobot-sim/`:
- With NVIDIA:

    ```bash
    rocker --home --network=host --user --volume ${PWD}:/tiago_public_ws/src/tiago-2021-accessrobot-sim --x11 --privileged ghcr.io/roboticslab-uc3m/tiago-2021-accessrobot-sim:latest --nvidia
    ```

- Else:

    ```bash
    rocker --home --network=host --user --volume ${PWD}:/tiago_public_ws/src/tiago-2021-accessrobot-sim --x11 --privileged ghcr.io/roboticslab-uc3m/tiago-2021-accessrobot-sim:latest --devices /dev/dri/card0
    ```

Make sure you ran the previous line from within `./tiago-2021-accessrobot`:
```bash
[ -f /tiago_public_ws/src/tiago-2021-accessrobot-sim/README.md ] && echo "OK!" || echo "FAIL! Make sure you ran the previous line from within 'tiago-2021-accessrobot-sim/'"
```

Move `tiago_data_generation` to `tiago_public_ws/src`, then:
```bash
cd /tiago_public_ws
sudo catkin build
source /tiago_public_ws/devel/setup.bash
roslaunch tiago_data_generation tiago_gazebo.launch
```

Simulation launches without visualization and writes a csv file. The structure of the csv depends on the `orient` argument in `datagen_controller.yaml`.
If `orient = "quaternion"`, the structure is:
```csv
arm_1_joint, arm_2_joint, arm_3_joint, arm_4_joint, arm_5_joint, arm_6_joint, arm_7_joint, end_effector_x, end_effector_y, end_effector_z, end_effector_quaternion_x, end_effector_quaternion_y, end_effector_quaternion_z, end_effector_quaternion_w
```
If `orient = "matrix"`, the structure is:
```csv
arm_1_joint, arm_2_joint, arm_3_joint, arm_4_joint, arm_5_joint, arm_6_joint, arm_7_joint, end_effector_x, end_effector_y, end_effector_z, end_effector_rotation_matrix_00, end_effector_rotation_matrix_01, end_effector_rotation_matrix_02, end_effector_rotation_matrix_10, end_effector_rotation_matrix_11, end_effector_rotation_matrix_12, end_effector_rotation_matrix_20, end_effector_rotation_matrix_21, end_effector_rotation_matrix_22
```
In any other case, it is:
```csv
arm_1_joint, arm_2_joint, arm_3_joint, arm_4_joint, arm_5_joint, arm_6_joint, arm_7_joint, end_effector_x, end_effector_y, end_effector_z
```

The file can be found in `/home/username/.ros` and its name can be set from `datagen_controller.yaml`, as well as the number of points to be generated.

## Preprocessing

Run `data_preprocessing.py` from tiago_data_generation scripts. This creates four files `x_train`, `x_test`, `y_train` and `y_test` in the data folder. This scripts needs 3 parameters, namely:
- `--norm`: chooses normalization type: 0 for standardization, 1 for min-max normalization, other for none
- `--orient`: chooses whether orientation data is used: 1 for pose, 0 for just position
- `--name`: choose name for split data files.

For example, running
```bash
rosrun tiago_data_generation data_preprocessing.py --norm=1 --orient=1 --name='name'
```
will produce files named "x_train_name.npy", "x_test_name.npy", etc, which will contain normalized pose inputs (position+orientation) and normalized joint outputs.