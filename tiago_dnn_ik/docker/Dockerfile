FROM tensorflow/tensorflow:2.14.0-gpu
# later tensorflow versions give forward compatibility error

# dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 graphviz git

# python dependencies
RUN pip install -q -U \
pyyaml pydot wandb keras-tuner pandas scikit-learn

# python fk
RUN apt install -y --no-install-recommends liboctomap-dev libfcl-dev
RUN git clone https://github.com/jdj2261/pykin.git
RUN cd pykin && python3 setup.py install
RUN mkdir /usr/local/lib/python3.11/dist-packages/pykin-1.6.0-py3.11.egg/pykin/assets/urdf/tiago/
RUN mkdir /usr/local/lib/python3.11/dist-packages/pykin-1.6.0-py3.11.egg/pykin/assets/urdf/tiago/tiago/
COPY data/urdf/ /usr/local/lib/python3.11/dist-packages/pykin-1.6.0-py3.11.egg/pykin/assets/urdf/tiago/
COPY data/urdf/meshes /usr/local/lib/python3.11/dist-packages/pykin-1.6.0-py3.11.egg/pykin/assets/urdf/tiago/tiago/meshes/

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all