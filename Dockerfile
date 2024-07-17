# ubuntu 22.04.3 LTS
FROM ubuntu:22.04

# download and install miniconda
RUN apt-get update && apt-get install wget ffmpeg libsm6 libxext6 -y

# install build essentials
RUN apt-get install -y build-essential

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
RUN bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p /miniconda3
RUN rm Miniconda3-py39_4.10.3-Linux-x86_64.sh

# set path to conda
ENV PATH="/miniconda3/bin:${PATH}"

# install python
RUN conda install -y python=3.9

# install pip
RUN conda install -y pip

# install dependencies
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install monai[all]==0.9.0

# copy requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# expose tensorboard port
# EXPOSE 6006

# entrypoint
CMD ["bash"]

# build docker image
# docker build -t atlas:latest .

# run interactive shell in docker container
# docker run -it --shm-size=32GB --gpus all -v /home/kaisar:/home/kaisar -p 6006:6006 --entrypoint /bin/bash atlas:latest

# run docker that calls python with a path to a script
# docker run -it --shm-size=32GB --gpus all -v /home/kaisar:/home/kaisar -p 6006:6006 atlas:latest python /home/kaisar/AbdomenAtlas/




