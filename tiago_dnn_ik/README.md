
## Step 1: Data generation
To generate the data using gazebo, see instructions [here](tiago_data_generation/gazebo_data_generation/datagen_info.md).

To use pykin, build first the docker image from the `tiago_dnn_ik` folder with:
```bash
bash docker/build.sh
```
Then, run the image with visualization using:
```bash
bash docker/run.sh 1
```
or without it
```bash
bash docker/run.sh
```
Urdf and meshes for TIAGo must be in `data/urdf` folder before building the container. Current urdf comes from [here](https://gitioc.upc.edu/rostutorials/ktmpb/-/blob/d9caa6ef59ffe14d557de47f60590e62d4cf4efa/demos/models/robots/tiago/)

To generate the data, run `pykin_generation.py` from tiago_data_generation scripts. This creates a csv file with the information needed for training.
This scripts needs 3 parameters, namely:
- `--rob`: chooses the robot urdf (tiago or dual)
- `--num`: chooses number of samples to generate
- `--file`: chooses resulting csv filename
- `--orient`: chooses orientation (quaternion, matrix or none)

For example, running
```bash
python3 tiago_data_generation/pykin_generation.py --rob='tiago' --num=100 --file='name' --orient='quaternion'
```
will produce a file named "name.csv" in the `data/pykin` folder that contains 100 samples of the form (joint position, ee position, ee quaternion)

## Step 2: Data preprocessing
To preprocess data generated using gazebo, see instructions [here](tiago_data_generation/datagen_info.md).

To use pykin, run `pykin_preprocessing` from tiago_data_generation scripts in the docker container. This creates four files `x_train`, `x_test`, `y_train` and `y_test` in the corresponding data pykin folder, times the number of curriculums. This scripts needs 4 parameters, namely:
- `--file`: chooses input csv filename
- `--norm`: chooses normalization type: 0 for standardization, 1 for min-max, 2 for max-absolute, 3 for robust, other for none
- `--name`: chooses name for split data files
- `--curr`: chooses number of curriculums generated for the data

For example, running
```bash
python3 tiago_data_generation/pykin_preprocessing.py --file='file' --norm=1 --name='name' --curr=1
```
will read "data/pykin/file.csv" and produce files named "x_train_curr1.npy", "x_test.npy", etc, in the "name" folder, which will contain normalized pose or position inputs (depending on what the csv contains) and normalized joint outputs. It will also create a "data_stats.yaml" file which contains min, max, std and mean for every column of data, among other information.

To change the normalization type of an already split dataset and keep the train-test-validation separation, run
```bash
python3 tiago_data_generation/change_normalization.py --file='file' --norm=1
```
where file is the one passed as 'name' to preprocessing and norm is the one to change to, following the same order as before.

## Step 3: Model building and training
To build and train the DNN models, tensorflow is needed, so build and run the docker container. 

To run the training for a specific model, run the `training.py` script specifying the name of the model. For example, to run the simple_mlp model training, do:
```bash
python3 training.py --name "SimpleMlp"
```

Parameters can be modified by changing the config yaml in the corresponding directory, as well as training and validation files, and wether he resulting trained model is saved in a .keras file or not.

Hyperparameter tuning can be performed by running the corresponding script with python for each directory. Additional arguments may be neccesary if more than one model is available in the selected folder.

The evolution of the different metrics can be followed in [Weights & Biases](https://wandb.ai/100518472/tiago_ik?nw=nwuser100518472).
