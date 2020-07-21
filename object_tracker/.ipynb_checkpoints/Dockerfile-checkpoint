FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget nginx sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
  
RUN pip install sagemaker sagemaker_inference boto3 
# needed for debug purposes only

  
############# D2 section ##############

# installing dependencies for D2 https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

ENV FORCE_CUDA="1"
# Build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta" 

# Build D2 from latest sources
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'


############ tracking_wo_bnw ##############

RUN git clone https://github.com/iprivit/tracking_wo_bnw
RUN cd tracking_wo_bnw && pip install -r requirements.txt && pip install -e .