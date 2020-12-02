# using Sagemaker PyTorch container as base image
# https://github.com/aws/sagemaker-pytorch-container/blob/master/docker/1.4.0/py3/Dockerfile.gpu
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.5.1-gpu-py36-cu101-ubuntu16.04

############# Installing latest builds ############

# This is to fix issue: https://github.com/pytorch/vision/issues/1489
RUN pip install --upgrade --force-reinstall torch==1.5.1 torchvision==0.6.1 cython
# RUN pip install torchvision==0.7.0

############# D2 section ##############

# installing dependecies for D2 https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore' 

ENV FORCE_CUDA="1"
# Build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta" 

# Build D2 from latest sources
# RUN pip install 'git+https://github.com/zhanghang1989/detectron2-ResNeSt.git'
# https://github.com/zhanghang1989/detectron2-ResNeSt.git
# https://github.com/facebookresearch/detectron2.git 
# latest version requires torch.jit.is_tracing which is only in torch 1.6.0

# Set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"
# set location of training datasetm, Sagemaker containers copy all data from S3 to /opt/ml/input/data/{channels}
ENV DETECTRON2_DATASETS="/opt/ml/input/data/train"

############# SageMaker section ##############

COPY container_training/plain_train_net.py /opt/ml/code/plain_train_net.py
# COPY container_training/ped_mask_rcnn_R_50_FPN_training_acc_test.yaml /opt/ml/code/ped_mask_rcnn_R_50_FPN_training_acc_test.yaml
WORKDIR /opt/ml/code

# cloning D2 to code dir as we need access to default congigs
RUN git clone 'https://github.com/facebookresearch/detectron2.git'
RUN cd detectron2 && git checkout be792b9 
RUN python -m pip install -e detectron2

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
# ENV SAGEMAKER_PROGRAM plain_train_net.py

WORKDIR /

# Starts PyTorch distributed framework
ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
