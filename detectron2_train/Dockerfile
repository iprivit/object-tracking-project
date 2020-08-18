FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.5.1-gpu-py36-cu101-ubuntu16.04
############# Installing latest builds ############

# This is to fix issue: https://github.com/pytorch/vision/issues/1489
RUN pip install --upgrade --force-reinstall torch torchvision cython
RUN pip install sagemaker sagemaker_inference boto3 # needed for debug purposes only

############# D2 section ##############

# installing dependencies for D2 https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install 'git+https://github.com/facebookresearch/fvcore'

ENV FORCE_CUDA="1"
# Build D2 only for Volta architecture - V100 chips (ml.p3 AWS instances)
ENV TORCH_CUDA_ARCH_LIST="Volta" 

# Build D2 from latest sources
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
COPY detectron2 /opt/ml/detectron2
# RUN pip install -e detectron2_repo

WORKDIR /opt/ml
#COPY model_final.pth /opt/ml/model/model_final.pth
# COPY model_final_5700.pth /opt/ml/model_final_5700.pth
# COPY model_final_13000.pth /opt/ml/model_final_13000.pth
COPY detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml /opt/ml/mask_rcnn_R_50_FPN_inference_acc_test.yaml
COPY detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml /opt/ml/mask_rcnn_R_50_FPN_3x.yaml
COPY detectron2/configs/Base-RCNN-FPN.yaml /opt/ml/Base-RCNN-FPN.yaml

# Set a fixed model cache directory. Detectron2 requirement
ENV FVCORE_CACHE="/tmp"
