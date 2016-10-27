FROM nvidia/cuda:7.5-cudnn4-devel

# Setup proxy
# ENV http_proxy
# ENV https_proxy

# Arguments
ARG CAFFE_VERSION=master
ARG TENSORFLOW_VERSION=0.11.0rc0
ARG TENSORFLOW_ARCH=gpu

# Install dependencies
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libatlas-base-dev \
    libboost-all-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-pip \
    python-numpy \
    python-scipy \
    libssl-dev \
    libffi-dev \  
    && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install pip --upgrade \
                pyOpenSSL \
                ndg-httpsclient \
                pyasn1

# Install Caffe
ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# Cloning & building caffe with gpu mode
RUN git clone -b ${CAFFE_VERSION} --depth 1 https://github.com/BVLC/caffe.git . && \
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 .. && \
    make -j"$(nproc)"

# Setup environment
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

# Overwriting $CAFFE_ROOT/python/caffe/io.py
COPY io.py $CAFFE_ROOT/python/caffe

# For "Error: libdc1394 error: Failed to initialize libdc1394"
#RUN sh -c 'ln -s /dev/null /dev/raw1394'; bash

# Install Chainer and Tensorflow, Keras
RUN pip --no-cache-dir install \
    chainer \
    https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_ARCH}/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl \
    keras

# This configuration avoid issue (https://github.com/tensorflow/tensorflow/issues/2626).
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Install python library
RUN pip --no-cache-dir install --upgrade \
        jupyter \
        matplotlib \
        scikit-learn \
        Pillow \
        scikit-optimize

# Generate configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/opt/notes'" >> /root/.jupyter/jupyter_notebook_config.py

ADD start.sh /usr/local/bin/start.sh
RUN chmod u+x /usr/local/bin/start.sh
EXPOSE 8888 6006
CMD ["/usr/local/bin/start.sh"]
