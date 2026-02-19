FROM rwthika/ros-ml

# dependencies
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    libsm6 libxext6 libxrender-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN echo "**** Installing Python ****" && \
    add-apt-repository ppa:deadsnakes/ppa &&  \
    apt-get install -y build-essential python3.9 python3.9-dev python3-pip && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm -rf /var/lib/apt/lists/*

# python dependencies
RUN pip install -q -U \
pandas numpy==1.23.5

RUN pip install --upgrade tensorflow==2.14.0

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all