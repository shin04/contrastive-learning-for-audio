#!/bin/bash

echo "meke need directories..."
mkdir dataset log models results
mkdir dataset/audioset dataset/esc dataset/generated_dataset dataset/hdf5
mkdir log/esc log/tensorboard log/training
echo "complete"

# echo "generate virtual env..."
# python3 -m venv cl_env
# source cl_env/bin/activate/
# pip install --upgrade pip
# pip install -r requirements.txt
# pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# echo "complete"

echo "setup complete"
echo "You must make .env file!!!"