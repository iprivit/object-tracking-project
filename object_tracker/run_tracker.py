import torch
from torch import nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from detectron2.engine.defaults import DefaultPredictor, DefaultTrainer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode

from tracktor.frcnn_fpn import FRCNN_FPN, MRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker, IJP_tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
from torchvision.ops.boxes import clip_boxes_to_image, nms

from PIL import Image
import itertools
import logging
from glob import glob
from tqdm import tqdm
import time, os, json
import pandas as pd
import boto3
import numpy as np
import matplotlib.patches as patches
from scipy.spatial import distance
import cv2
import sys
from collections import OrderedDict
from collections import deque
from matplotlib import patches

from scipy.optimize import linear_sum_assignment
import argparse
import subprocess
import datetime
from matplotlib import pyplot as plt
from matplotlib import patches
import warnings

warnings.filterwarnings(action='ignore')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sns = boto3.client('sns')
s3 = boto3.client('s3')
a2i = boto3.client('sagemaker-a2i-runtime')
sagemaker_client = boto3.client('sagemaker')


config_file = '/home/ubuntu/code/detectron2-ResNeSt/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml'
# config_file = '/home/ubuntu/code/detectron2-ResNeSt/configs/COCO-Detection/helmet_faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x.yaml'

torch.backends.cudnn.deterministic = True
device="cuda:0"

# obj_detect = FRCNN_FPN(num_classes=2) # may need to train this specific network using detectron2 
# # it looks like there are custom helper functions in this class 
# obj_detect.load_state_dict(torch.load( 'output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model', #_config['tracktor']['obj_detect_model'],
#                            map_location=device))#lambda storage, loc: storage))

# obj_detect.eval()
#obj_detect.cuda()

def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0] - w1 / 2.0, box2[0] - w2 / 2.0)
        x2_max = max(box1[0] + w1 / 2.0, box2[0] + w2 / 2.0)
        y1_min = min(box1[1] - h1 / 2.0, box2[1] - h2 / 2.0)
        y2_max = max(box1[1] + h1 / 2.0, box2[1] + h2 / 2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea / uarea)



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model", default='helmet_tracking/model_final_13000.pth', type=str, #required=True,
                        help="path to base D2 model or s3 location")
    parser.add_argument("--reid_model", default='helmet_tracking/ResNet_iter_25137.pth', type=str, #required=True, 
                        help="path to reid model or s3 location")
    parser.add_argument("--output_dir", default='/home/model_results', type=str, #required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--bucket", default='privisaa-bucket-virginia', type=str)
    parser.add_argument("--img_paths",
                        default='nfl-data/live_video',
                        type=str,
                        #required=True,
                        help="path to images or image location in s3")
    parser.add_argument("--conf_thresh", default=.5, type=float, #required=True,
                        help="base D2 model")   
    parser.add_argument("--use_mask", default=0, type=int)
#     parser.add_argument("--video", default=None, type=str, required=True, help="path to video for tracking job")
    parser.add_argument("--d2_config", default='/home/detectron2-ResNeSt/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml', type=str, help="Detectron2 config file")
    args = parser.parse_args()
    
    #COCO-Detection/faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x.yaml'
    confidence_threshold= .55
    cfg = get_cfg()
    config_file = args.d2_config
    model_pth = args.model
    reid_pth = args.reid_model
    
    try:
        os.mkdir('/home/video')
    except:
        pass
    
    if os.path.exists(args.model)==False:
        print('\n')
        print('downloading d2 model from s3')
        s3.download_file(args.bucket, args.model, '/home/d2_final.pth')
        model_pth = '/home/d2_final.pth'
    if os.path.exists(args.reid_model)==False:
        print('\n')
        print('downloading reid model from s3')
        s3.download_file(args.bucket, args.reid_model, '/home/reid_final.pth')
        reid_pth = '/home/reid_final.pth'    
    if os.path.exists(args.img_paths)==False:
        objs = s3.list_objects(Bucket='privisaa-bucket-virginia', Prefix=args.img_paths)['Contents']
        keys = []
        for key in objs:
            if key['Size']>0:
                keys.append(key['Key'])
        folders = []
        for key in keys:
            folders.append(key.split('/')[-2])
        folder = list(np.unique(folders))[-1]
        print('\n')
        print('Loading images from this video: ',folder)
        for key in tqdm(keys):
            if key.split('/')[-2]==folder:
                s3.download_file(args.bucket, key, f"/home/video/{key.split('/')[-1]}")

    cfg.merge_from_file(config_file)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.conf_thresh
    cfg.MODEL.WEIGHTS = model_pth #'/home/ubuntu/finetuned-fasterrcnn-cascade-d2-resnest-13000imgs-005lr-1class-tune2/model_final.pth'
    #finetuned-fasterrcnn-cascade-d2-resnest-13000imgs-02lr/model_final.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_thresh
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.conf_thresh
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    model = DefaultPredictor(cfg)
    #model.model.load_state_dict(torch.load('finetuned-detectron2-maskrcnn_13000imgs_03lr_3rdRun/model_final.pth')['model'])
    eval_results = model.model.eval()#                                        finetuned-detectron2-maskrcnn_fixed/model_final.pth')['model'])
    # 'finetuned-detectron2-maskrcnn_fixed/model_final.pth'
    # finetuned-detectron2-maskrcnn_5700imgs_025lr_furthertuned/model_final.pth
    
    if args.use_mask==0:
        obj_detect = MRCNN_FPN(model,num_classes=1)
    else:
        obj_detect = FRCNN_FPN(model, num_classes=1)

    #obj_detect.to(device)

    # reid
    reid_network = resnet50(pretrained=False, output_dim=128)  # need to insert dictionary here 
    reid_network.load_state_dict(torch.load( reid_pth, #tracktor['reid_weights'],'/home/ubuntu/code/tracking_wo_bnw/output/tracktor/reid/test/ResNet_iter_25137.pth'
                                            map_location=device)) #lambda storage, loc: storage))
    reid_network.eval()
    #reid_network.cuda()
    reid_network.to(device)

    tracker={
        # FRCNN score threshold for detections
        'detection_person_thresh': 0.5,
        # FRCNN score threshold for keeping the track alive
        'regression_person_thresh': 0.5,
        # NMS threshold for detection
        'detection_nms_thresh': 0.3,
        # NMS theshold while tracking
        'regression_nms_thresh': 0.6,
        # motion model settings
        'motion_model':{
          'enabled': True,
          # average velocity over last n_steps steps
          'n_steps': 2,
          # if true, only model the movement of the bounding box center. If false, width and height are also modeled.
          'center_only': True},
        # DPM or DPM_RAW or 0, raw includes the unfiltered (no nms) versions of the provided detections,
        # 0 tells the tracker to use private detections (Faster R-CNN)
        'public_detections': False,
        # How much last appearance features are to keep
        'max_features_num': 20,
        # Do camera motion compensation
        'do_align': False,
        # Which warp mode to use (cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE, ...)
        'warp_mode': str(cv2.MOTION_EUCLIDEAN),
        # maximal number of iterations (original 50)
        'number_of_iterations': 500,
        # Threshold increment between two iterations (original 0.001)
        'termination_eps': 0.00001,
        # Use siamese network to do reid
        'do_reid': True,
        # How much timesteps dead tracks are kept and cosidered for reid
        'inactive_patience': 15,
        # How similar do image and old track need to be to be considered the same person
        'reid_sim_threshold': 20.0,
        # How much IoU do track and image need to be considered for matching
        'reid_iou_threshold': 0.05
      }

    tracker = IJP_tracker(obj_detect, reid_network, tracker)
    # tracker = Tracker(obj_detect, reid_network, tracker)

    def transform_img(i ,pth = '/home/ubuntu/videos/'):
        if(i<10):
            ind = f'0000{i}'
        elif(i<100):
            ind = f'000{i}'
        else:
            ind = f'00{i}'
        frame = Image.open(f'{pth}split_frames_{ind}.jpg')
        frame_ten = torch.tensor(np.reshape(np.array(frame),(1,3,720,1280)), device=device, dtype=torch.float32)
        blob = {'img':frame_ten}
        return blob
    # cannot make frame size assumption...

    # need to make this an argparse argument 
    pth = '/home/video/'
#     vid_name = args.video #'57675_003286_Endzone'
    img_paths = glob(f'{pth}/*') # {vid_name}
    img_paths.sort()
#     print(img_paths)

    tracker.reset()

#     results = []
#     print('Starting tracking!')
#     for i in tqdm(range(1,len(img_paths)+1)):
#         #tracker.reset()
#         blob = transform_img(i ,pth=pth)
#         with torch.no_grad():
#             tracker.step(blob)
            
    track_dict = {}

    print("\n")
    print("###########################################")
    print("############## BEGIN TRACKING #############")
    print("###########################################")
    print('\n')

    for i in tqdm(range(1,len(img_paths)+1)):
        track_dict[i] = {}
        blob = transform_img(i,pth=pth)
        with torch.no_grad():
            tracker.step(blob)
        for tr in tracker.tracks:
            track_dict[i][tr.id] = tr.pos[0].detach().cpu().numpy()
            
    iou_dict = {}
    iou_dict2 = {}
    new_tracks = {}
    missing_tracks = {}

    for track in tqdm(track_dict):
        if track==1: # set dictionaries with 1st frame
            new_tracks[track] = list(track_dict[track].keys())
            missing_tracks[track] = set()
            iou_dict[track] = {}
            iou_dict[track] = {}
            for tr in track_dict[track]:
                iou_dict[track][tr] = {}
        else:
            new_tracks[track] = set(list(track_dict[track].keys()))  - set(new_tracks[1]) # - set(list(track_dict[track-1].keys()))
            missing_tracks[track] = set(new_tracks[1]) - set(list(track_dict[track-1].keys()))
            iou_dict[track] = {}
            iou_dict2[track] = {}
            for tr in track_dict[track]:
                iou_dict[track][tr] = {}
                iou_dict2[track][tr] = {}
                for t in track_dict[track-1]:
                    iou = bbox_iou(track_dict[track][tr], track_dict[track-1][t])
                    if iou>0.001:
                        iou_dict[track][tr][t] = iou
                if track>2:
                    for t in track_dict[track-2]:
                        iou = bbox_iou(track_dict[track][tr], track_dict[track-2][t])
                        if iou>0.001:
                            iou_dict2[track][tr][t] = iou   
                            
    tracks_to_delete = {}
    tracks_to_change = {}                        
    momentum_dict = {}
    for track in track_dict:
        if track==1:
            for tr in track_dict[track]:
                momentum_dict[tr] = 0 # initialize momentum dict 
        elif len(track_dict[track])>22:  # if there are more than 22 annotations
            tracks_to_delete[track] = []
            for tr in track_dict[track]:
                try:
                    momentum_dict[tr] +=1
                except:
                    momentum_dict[tr] = 0
            for ind in iou_dict[track]:
                if (iou_dict[track][ind]=={})&(ind>22):  # need to adjust this, right now just looking if there is ANY IoU
                    tracks_to_delete[track].append(ind)
        else: # if less than 22 annotations
            tracks_to_change[track] = {}
            if new_tracks[track]!=set():  # if there are new tracks, check them
                for tr in track_dict[track]: # update momentum dict
                    try:
                        momentum_dict[tr] +=1
                    except:
                        momentum_dict[tr] = 0
                for newt in new_tracks[track]: # cycle through new tracks 
    #                 print('For track ',track,"and ID ",newt,iou_dict[track][newt])
                    if newt>22:  # if there is a new track and it's greater than 22, change it, figure out what to change to
                        if (missing_tracks[track]!=set())|(missing_tracks[track-1]!=set()): # if there are missing tracks, cycle through them and compare
                            mis_iou = {}    
                            if missing_tracks[track]!=set():
                                for mis in missing_tracks[track]:
                                    try:
    #                                     iou = bbox_iou(track_dict[track-1][mis], track_dict[track][newt]) 
                                        dist = distance.euclidean(track_dict[track-1][mis], track_dict[track][newt])
                                    except:
                                        try:
    #                                         iou = bbox_iou(track_dict[track-2][mis], track_dict[track][newt]) 
                                            dist = distance.euclidean(track_dict[track-2][mis], track_dict[track][newt])
                                        except:
                                            try:
    #                                             iou = bbox_iou(track_dict[track-3][mis], track_dict[track][newt]) 
                                                dist = distance.euclidean(track_dict[track-3][mis], track_dict[track][newt])
                                            except:
                                                pass 

                                    mis_iou[mis] = dist
    #                                 tracks_to_change[track][newt] = 
                                    tracks_to_change[track][newt] = mis_iou
    #                         if missing_tracks[track-1]!=set():
    #                             for mis in missing_tracks[track-1]:
    #                                 iou = bbox_iou(track_dict[track-2][mis], track_dict[track][newt]) 
    #                                 mis_iou[mis] = iou
    #                                 tracks_to_change[track][newt] = mis_iou

    #                         try:
    #                             if max(tracks_to_change[track][newt].values())>.2:
                    #                 for t in tracks_to_change[trc][tr]:
                                to_ind = np.argmin(list(tracks_to_change[track][newt].values()))
                                to_id = list(tracks_to_change[track][newt].keys())[to_ind]
                                to_pos = track_dict[track][newt]
                                del track_dict[track][newt]
                                track_dict[track][to_id] = to_pos  

    # need to send results to s3 
    result = tracker.get_results()
    file = 'tracking_results.json'
    tracking_dict = {}
    for res in result:
        tracking_dict[res] = {}
        for r in result[res]:
            tracking_dict[res][r] = list(result[res][r][0:4])

    with open(file,'w') as f:
        json.dump(tracking_dict,f)
#     now = str(datetime.datetime.now()).replace(' ','').replace(':','-')
    k = f'nfl-data/tracking_results_{folder}.json'
    s3.upload_file(Filename=file, Bucket=args.bucket, Key=k)
    print(f'Tracking finished and results saved to: s3://{args.bucket}/{k}')
    
    os.makedirs('/home/labeled_frames')
    # create labeled images
    print('\n')
    print("###########################################")
    print("############ Generating Video! ############")
    print("###########################################")
    print('\n')
    print('...')
    for j,pth in tqdm(enumerate(img_paths)):
        fig,ax = plt.subplots(1, figsize=(24,14))

        img = Image.open(pth)
        # Display the image
        ax.imshow(np.array(img))

        # Create a Rectangle patch
        label_list = {}
        for r in track_dict[j+1]:
            try:
                res = track_dict[j+1][r]
                label_list[r] = res
            except:
                pass
        for i,r in enumerate(label_list):
            labs = label_list[r]
            rect = patches.Rectangle((labs[0], labs[1]),labs[2]-labs[0],labs[3]-labs[1] ,linewidth=1,edgecolor='r',facecolor='none') # 50,100),40,30
            ax.add_patch(rect)
            plt.text(labs[0]-10, labs[1]-10, f'H:{r}', fontdict=None)

        plt.savefig(f"/home/labeled_frames/{pth.split('/')[-1].replace('.jpg','.png')}")
        plt.close()
    # create video of labels 
    os.system('ffmpeg -r 15 -f image2 -s 1280x720 -i /home/labeled_frames/split_frames_%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p /home/labeled_frames.mp4')
    k = f'nfl-data/tracking_results_{folder}.mp4'
    s3.upload_file(Filename='/home/labeled_frames.mp4', Bucket=args.bucket, Key=k)
    print('\n')
    print(f'Video uploaded to: s3://{args.bucket}/{k}')

    # for launching A2I job set a conditional here 
    s3_fname = f's3://{args.bucket}/{k}'
    workteam = 'arn:aws:sagemaker:us-east-1:209419068016:workteam/private-crowd/ijp-private-workteam'
    flowDefinitionName = 'ijp-video-flow-official'
    humanTaskUiArn = 'arn:aws:sagemaker:us-east-1:209419068016:human-task-ui/ijp-video-task3'

    #'s3://privisaa-bucket-virginia/nfl-data/nfl-frames/nfl-video-frame00001.png'
#     create_workflow_definition_response = sagemaker_client.create_flow_definition(
#             FlowDefinitionName= flowDefinitionName,
#             RoleArn= role,
#             HumanLoopConfig= {
#                 "WorkteamArn": workteam,
#                 "HumanTaskUiArn": humanTaskUiArn,
#                 "TaskCount": 1,
#                 "TaskDescription": "Identify if the labels in the video look correct.",
#                 "TaskTitle": "Video classification a2i demo"
#             },
#             OutputConfig={
#                 "S3OutputPath" : OUTPUT_PATH
#             }
#         )
#     flowDefinitionArn = create_workflow_definition_response['FlowDefinitionArn']
    
    inputContent = {
                "initialValue": .2,
                "taskObject": s3_fname # the s3 object will be passed to the worker task UI to render
            }
    
    now= str(datetime.datetime.now()).replace(' ','-').replace(':','-').replace('.','-')
    response = a2i.start_human_loop(
        HumanLoopName=f'ijp-video-{now}',
        FlowDefinitionArn=flowDefinitionArn,
        HumanLoopInput={
            "InputContent": json.dumps(inputContent)
        },
        DataAttributes={
            'ContentClassifiers': ['FreeOfPersonallyIdentifiableInformation'
            ]
        }
    )
    print(f'Launched A2I loop ijp-video-{now}')
    
    sns.publish(TopicArn='arn:aws:sns:us-east-1:209419068016:ijp-topic',Message=f'Your video inference is done! You can find the output here: s3://{args.bucket}/{k}',Subject='Video labeling')
    
    
if __name__ == "__main__":
    main()
#     dllogger.flush()
