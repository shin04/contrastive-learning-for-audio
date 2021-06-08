FROM nvidia/cuda:11.0-devel-ubuntu20.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y libsndfile1

WORKDIR /ml
COPY ./requirements.txt /ml
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html