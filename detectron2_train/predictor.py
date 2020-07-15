import os
import json
import pickle
import sys
import signal
import traceback
import re
import flask
import numpy as np
import boto3
import tarfile
import logging

import torch
from pathlib import Path
import warnings

from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
import detectron2.data.transforms as T
from detectron2.config import get_cfg

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

prefix = "/opt/ml/"
PATH = Path(os.path.join(prefix, "model"))
PRETRAINED_PATH = Path(os.path.join(prefix, "code"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# with open('hyperparameters.json') as f:
#     params = json.load(f)  
# bucket = params['save_to_s3']

model_path = '/opt/ml/model/model_final.pth' #"pytorch_model.bin"
config_path = '/home/appuser/detectron2_repo/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_predictor_model(cls):

        cfg = get_cfg()

        cfg.merge_from_file(config_path) # get baseline parameters from YAML config
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
#         cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        model = DefaultPredictor(cfg)
        model.model.load_state_dict(torch.load('/home/appuser/detectron2_repo/model_final.pth')["model"])
#         logger.info(cfg)
        model.model.eval()
        
#         model.to(device)
        cls.model = model

        return cls.model

    @classmethod
    def predict(cls, img, 
               ):
        """For the input, do the predictions and return them.
        Args:
            input: image"""
        predictor_model = cls.get_predictor_model()
        
        # run prediction # DO YOU NEED TORCH.NO_GRAD() ????
        with torch.no_grad():
            prediction = predictor_model(img)
            
        # post-processing      
        return prediction

    @classmethod
    def searching_all_files(cls, directory: Path):
        file_list = []  # A list for storing files existing in directories

        for x in directory.iterdir():
            if x.is_file():
                file_list.append(str(x))
            else:
                file_list.append(cls.searching_all_files(x))

        return file_list


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = (
        ScoringService.get_predictor_model() is not None
    )  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json") # can change to x-npy


# @app.route("/execution-parameters", method=["GET"])
# def get_execution_parameters():
#     params = {
#         "MaxConcurrentTransforms": 3,
#         "BatchStrategy": "MULTI_RECORD",
#         "MaxPayloadInMB": 6,
#     }
#     return flask.Response(
#         response=json.dumps(params), status="200", mimetype="application/json"
#     )


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
#     data = None

    if flask.request.content_type == "application/json":
        print("calling json launched")
        data = flask.request.get_json(silent=True) # failing here, returning None
#         data = json.loads(data)
        print(data)

        img = data["img"] # failing to get image here 
        img = np.array(img)
        logger.info(img.shape)


    else:
        return flask.Response(
            response="This predictor only supports JSON data",
            status=415,
            mimetype="text/plain",
        )

    #print("Invoked with text: {}.".format(text.encode("utf-8")))

    # Do the prediction
    predictions = ScoringService.predict(img) 
    logger.info(predictions)

    result = json.dumps(predictions) # may need to fix this 

    return flask.Response(response=result, status=200, mimetype="application/json")