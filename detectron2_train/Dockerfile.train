FROM nvidia/cuda:10.1-cudnn7-devel

############# Installing latest builds ############

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget nginx sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# This is to fix issue: https://github.com/pytorch/vision/issues/1489
# RUN pip install --upgrade --force-reinstall torch torchvision cython
ENV PATH="/opt/ml/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py
    
############# D2 section ##############

RUN pip install tensorboard cython boto3 numpy==1.15.4 gevent pandas flask pathlib gunicorn tqdm  requests six ipdb h5py html2text nltk progressbar onnxruntime sagemaker sagemaker_inference pycocotools git+https://github.com/NVIDIA/dllogger
RUN pip install torch==1.5+cu101 torchvision==0.6+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

# installing dependencies for D2 https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile

ENV FORCE_CUDA="1"
# Build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta" 

# Build D2 from latest sources
# RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
COPY detectron2 /opt/ml/detectron2
RUN pip install -e /opt/ml/detectron2

WORKDIR /opt/ml
# #COPY model_final.pth /opt/ml/model/model_final.pth
# COPY model_final_5700.pth /opt/ml/model_final_5700.pth
# COPY model_final_13000.pth /opt/ml/model_final_13000.pth
COPY detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml /opt/ml/mask_rcnn_R_50_FPN_inference_acc_test.yaml
COPY detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml /opt/ml/mask_rcnn_R_50_FPN_3x.yaml
COPY detectron2/configs/Base-RCNN-FPN.yaml /opt/ml/Base-RCNN-FPN.yaml
COPY train /opt/ml/.local/train

# Set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"


# # # need to put train command HERE
ENV PATH="/opt/ml/.local:${PATH}"
# /bin


# # # install dependencies
# # # See https://pytorch.org/ for other options if you use a different version of CUDA # remove user options --user
# RUN pip install tensorboard cython boto3 numpy==1.15.4 gevent flask pathlib gunicorn tqdm  requests six ipdb h5py html2text nltk progressbar onnxruntime sagemaker sagemaker_inference pycocotools git+https://github.com/NVIDIA/dllogger
# RUN pip install torch==1.5+cu101 torchvision==0.6+cu101 -f https://download.pytorch.org/whl/torch_stable.html


# # RUN pip install --upgrade --no-cache-dir pip \
# #  && pip install --no-cache-dir \

# # install detectron2
# #RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo

# # # set FORCE_CUDA because during `docker build` cuda is not accessible
# # # This will by default build detectron2 for all common cuda architectures and take a lot more time,
# # # because inside `docker build`, there is no way to tell which architecture will be used.
# ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
# ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
