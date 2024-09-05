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

## 4. Set up library path
Add the Conda environment's library path to LD_LIBRARY_PATH:

export LD_LIBRARY_PATH="/home/dnth/mambaforge-pypy3/envs/onnx-gpu/lib:$LD_LIBRARY_PATH"

Note: Adjust the path according to your Conda environment location.
