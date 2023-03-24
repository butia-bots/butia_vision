#!/bin/sh

[ $(which git) ] || sudo apt install git
pip install -r ./requirements.txt

mkdir -p ~/butia_ws/src
cd ~/butia_ws/src

# Clonando repositórios
git clone https://github.com/butia-bots/iai_kinect2.git
git clone https://github.com/butia-bots/libfreenect2.git

#cd ~/butia_ws/src/butia_vision

#if [[ $SHELL_TYPE == "bash" ]]; then
#    echo "export PYTHONPATH=/usr/local/lib/:$PYTHONPATH" >> ~/.bashrc
#    echo "export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH" >> ~/.bashrc
#    echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
#    source ~/.bashrc
#elif [[ $SHELL_TYPE == "zsh" ]]; then
#    echo "export PYTHONPATH=/usr/local/lib/:$PYTHONPATH" >> ~/.zshrc
#    echo "export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH" >> ~/.zshrc
#    echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.zshrc
#    source ~/.zshrc
#else
#    echo "Não foi possível determinar o tipo de shell"
#    exit
#fi
