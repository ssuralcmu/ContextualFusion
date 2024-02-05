FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda install python=3.8
RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack

RUN conda install -c conda-forge cudatoolkit-dev
RUN export FORCE_CUDA="1"
ENV FORCE_CUDA="1"
ENV CUDA_HOME="/usr/local/cuda-11.3"
#RUN export CUDA_HOME="/usr/local/cuda-11.3"
RUN echo $CONDA_PREFIX

#RUN export CUDA_HOME=$CONDA_PREFIX 
RUN echo $CUDA_HOME

RUN pip install --upgrade pip setuptools wheel

RUN pip install -U openmim
RUN mim install mmcv==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.1/index.html

RUN pip install mmdet==2.20.0

RUN pip install nuscenes-devkit
RUN pip install mpi4py==3.0.3
RUN pip install numba==0.48.0
RUN pip install setuptools==59.5.0

# RUN CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
# RUN echo $CUDA_VERSION
# RUN if [[ ${CUDA_VERSION} == 9.0* ]]; then \
#      ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;7.0+PTX" \
#      if [[ ${CUDA_VERSION} == 9.2* ]]; then \
#      ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0+PTX" \
#      if [[ ${CUDA_VERSION} == 10.* ]]; then \
#      ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX" \
#      if [[ ${CUDA_VERSION} == 11.0* ]]; then \
#      ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX" \
#      if [[ ${CUDA_VERSION} == 11.* ]]; then \
#      ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX" \
#      else \
#      echo "unsupported cuda version." \
#      fi

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0+PTX"

RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/1.10.1/index.html

RUN pip install --upgrade numpy==1.19.5

RUN pip install yapf==0.40.1
