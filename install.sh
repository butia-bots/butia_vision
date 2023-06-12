#!/bin/bash

cd ~/butia_ws/src/butia_vision

SHELL_TYPE=$(echo $SHELL | grep -oE '[^/]+/?$' | tr -d '/')

if [[ $SHELL_TYPE == "bash" ]]; then
    echo "export PYTHONPATH=/usr/local/lib/:$PYTHONPATH" >> ~/.bashrc
    echo "export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/lib/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/:/usr/local/lib/python3.8/dist-packages/nvidia/cuda_runtime/lib/:/usr/local/lib/python3.8/dist-packages/nvidia/cufft/lib/:/usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib/:/usr/local/lib/python3.8/dist-packages/nvidia/cusparse/lib/"
    source ~/.bashrc
elif [[ $SHELL_TYPE == "zsh" ]]; then
    echo "export PYTHONPATH=/usr/local/lib/:$PYTHONPATH" >> ~/.zshrc
    echo "export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH" >> ~/.zshrc
    echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.zshrc
    echo "export LD_LIBRARY_PATH=/usr/local/lib/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu:/usr/local/lib:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib/:/usr/local/lib/python3.8/dist-packages/nvidia/cuda_runtime/lib/:/usr/local/lib/python3.8/dist-packages/nvidia/cufft/lib/:/usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib/:/usr/local/lib/python3.8/dist-packages/nvidia/cusparse/lib/"
    source ~/.zshrc
else
    echo "Não foi possível determinar o tipo de shell"
    exit
fi
