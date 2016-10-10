FROM ubuntu:14.04

# Setup proxy
#ENV http_proxy
#ENV https_proxy

# Arguments
ARG CAFFE_VERSION=master
ARG TENSORFLOW_VERSION=0.8.0
ARG TENSORFLOW_ARCH=cpu
ARG KERAS_VERSION=1.0.3 

# Install dependencies
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    build-essential \
