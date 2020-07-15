import os
import torch
from detectron2.engine.defaults import DefaultPredictor, DefaultTrainer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode

#os.system('git clone https://github.com/facebookresearch/detectron2.git')

config_file = 'detectron2/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml'
confidence_threshold= .6
cfg = get_cfg()

cfg.merge_from_file(config_file)
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.DEVICE = 'cuda:0'


# do some kind of split based on a character that splits off the 3 inputs 
def input_fn(request_body, request_content_type):
    if(request_content_type =='application/json'):
        print('request_body: ',request_body)
        input_data = json.load(StringIO(request_body))
        print('input_data: ', input_data)
        context = input_data['img']


    elif(request_content_type == 'application/x-npy'):
        try:
            input_data = np.frombuffer(request_body, dtype=int)
        except:
            input_data = np.array(request_body, dtype=int)
        try:
            input_data = np.reshape(input_data,(256,256,3))
        except:
            input_data = np.reshape(input_data[16:],(256,256,3)) 
    return input_data

def predict_fn(input_data, model):
    # run prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.model.eval()
    with torch.no_grad():
        pred = model(img)

    return pred
    
def model_fn(model_dir):
    model = DefaultPredictor(cfg)
    try:
        model.model.load_state_dict(torch.load('/opt/ml/model/model_final.pth')['model']) # 
    except:
        model.model.load_state_dict(torch.load('/opt/ml/model/opt/ml/model/model_final.pth', map_location='cpu')["model"])
    return model