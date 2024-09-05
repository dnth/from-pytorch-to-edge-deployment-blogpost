# ONNX Runtime with CUDA Setup Guide

This guide provides steps to set up ONNX Runtime with CUDA support using Conda.

## 1. Install CUDA components
Install CUDA 12.2.2 and related tools:

conda install -y -c nvidia cuda=12.2.2 cuda-tools=12.2.2 cuda-toolkit=12.2.2 cuda-version=12.2 cuda-command-line-tools=12.2.2 cuda-compiler=12.2.2 cuda-runtime=12.2.2

## 2. Install cuDNN
Install cuDNN 9.2.1.18:

conda install cudnn==9.2.1.18

## 3. Install ONNX Runtime GPU
Install ONNX Runtime with GPU support:

pip install -U onnxruntime-gpu==1.19.2

## 4. Install TensorRT
Install TensorRT and its dependencies:

pip install tensorrt==10.1.0 tensorrt-cu12==10.1.0 tensorrt-cu12-bindings==10.1.0 tensorrt-cu12-libs==10.1.0

## 5. Set up library paths
Add the Conda environment's library path and TensorRT library path to LD_LIBRARY_PATH:

export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/onnx-gpu/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/onnx-gpu/lib/python3.11/site-packages/tensorrt_libs:$LD_LIBRARY_PATH"

Note: Adjust the paths according to your Conda environment location.
