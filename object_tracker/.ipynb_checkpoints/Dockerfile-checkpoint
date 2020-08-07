FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt-get update && apt-get install -y \
	ca-certificates python3-dev git wget ffmpeg nginx sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev && \
  rm -rf /var/lib/apt/lists/*
  
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py
    

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install   tensorboard sagemaker sagemaker_inference boto3 opencv-contrib-python
RUN pip install --upgrade --force-reinstall torch torchvision cython
  
############# D2 section ##############

# installing dependencies for D2 https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

ENV FORCE_CUDA="1"
# Build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta" 
ENV PATH="/bin:${PATH}"

RUN mkdir /home/model_results
RUN mkdir /home/video

# Build D2 from latest sources
RUN git clone https://github.com/zhanghang1989/detectron2-ResNeSt /home/detectron2-ResNeSt
RUN pip install -e /home/detectron2-ResNeSt

COPY serve /bin/serve
COPY run_tracker.py /home/run_tracker.py
COPY .aws /root/.aws

############ tracking_wo_bnw ##############

RUN git clone https://github.com/iprivit/tracking_wo_bnw
RUN cd tracking_wo_bnw && pip install -r requirements.txt && pip install -e .

# fix issues with opencv install
RUN sudo apt -y update && apt -y install libglib2.0-0 libsm6 libxrender1

RUN chmod +x /bin/serve


