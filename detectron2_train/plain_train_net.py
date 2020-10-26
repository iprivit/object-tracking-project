#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
#     CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper 
from detectron2.structures import BoxMode
import pandas as pd
import numpy as np
import itertools
import boto3
import subprocess
import argparse

# print('contents of opt/ml: ',subprocess.check_output(['ls', '/opt/ml']))
# print('contents of opt/ml/input/data/train: ',subprocess.check_output(['ls', '/opt/ml/input/data/train']))

s3 = boto3.client('s3')
# can you just put this in your train folder and it will get downloaded?


logger = logging.getLogger("detectron2")

df = pd.read_csv('/opt/ml/input/data/train/sagemaker_mot20_annotations.csv')
unique_files = df.file_name.unique()
train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.9), replace=False))
train_df = df[df.file_name.isin(train_files)]
test_df = df[~df.file_name.isin(train_files)]
classes = df.class_name.unique().tolist()
    
for d in ["train", "val"]:
    DatasetCatalog.register("pedmot20_" + d, lambda d=d: create_dataset_dicts(train_df if d == "train" else test_df, classes))
    MetadataCatalog.get("pedmot20_" + d).set(thing_classes=classes)
statement_metadata = MetadataCatalog.get("pedmot20_train")

# print('torch cuda version:', torch.version.cuda)

# s3.download_file(Bucket='privisaa-bucket-virginia', Key='nfl-data/sagemaker_helmet_annotations.csv', Filename='helmet_annotations.csv')
# df = pd.read_csv('helmet_annotations.csv')

# IMAGES_PATH = f'helmet_images'

# unique_files = df.file_name.unique()

# # need to have the same train and test files each time...
# train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.9), replace=False))
# train_df = df[df.file_name.isin(train_files)]
# test_df = df[~df.file_name.isin(train_files)]

def create_dataset_dicts(df, classes):
  dataset_dicts = []
  for image_id, img_name in enumerate(df.file_name.unique()):

    record = {}

    image_df = df[df.file_name == img_name]

    file_path = f'{img_name}' # {IMAGES_PATH}/
    record["file_name"] = file_path
    record["image_id"] = image_id
    record["height"] = int(image_df.iloc[0].height)
    record["width"] = int(image_df.iloc[0].width)

    objs = []
    for _, row in image_df.iterrows():

      xmin = int(row.x_min)
      ymin = int(row.y_min)
      xmax = int(row.x_max)
      ymax = int(row.y_max)

      poly = [
          (xmin, ymin), (xmax, ymin), 
          (xmax, ymax), (xmin, ymax)
      ]
      poly = list(itertools.chain.from_iterable(poly))

      obj = {
        "bbox": [xmin, ymin, xmax, ymax],
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [poly],
        "category_id": classes.index(row.class_name),
        "iscrowd": 0
      }
      objs.append(obj)

    record["annotations"] = objs
    dataset_dicts.append(record)
  return dataset_dicts

def _opts_to_list(opts):
    """
    This function takes a string and converts it to list of string params (YACS expected format). 
    E.g.:
        ['SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.9999'] -> ['SOLVER.IMS_PER_BATCH', '2', 'SOLVER.BASE_LR', '0.9999']
    """
    import re
    
    if opts!=None:
        list_opts = re.split('\s+', opts)
        return list_opts
    return ""

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    try:
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    except:
        evaluator_type = 'coco'
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


        
def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    checkpointer_spot = DetectionCheckpointer(
        model, '/opt/ml/checkpoints', optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    periodic_checkpointer_spot = PeriodicCheckpointer(
        checkpointer_spot, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )    

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
#     data_loader = build_detection_train_loader(cfg)
    data_loader = build_detection_train_loader(cfg,
#    mapper=DatasetMapper(cfg, is_train=True
#                         , augmentations=[
#        T.Resize((1024, 1024)),
#        T.RandomBrightness(.75,1.25),
#        T.RandomFlip(),
#        T.RandomSaturation(.75,1.25)
#    ]
                       )
#     )
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()
            if iteration%500==0:
                try:
                    torch.save(model.state_dict(), f'{cfg.OUTPUT_DIR}/model_{iteration}.pth')
                except:
                    print('save failed')

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
            periodic_checkpointer_spot.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    if (sm_args.spot_ckpt!='')&(sm_args.spot_ckpt is not None):
        os.system(f"aws s3 cp {sm_args.spot_ckpt} /opt/ml/checkpoints/{sm_args.spot_ckpt.split('/')[-1]}")
        cfg.MODEL.WEIGHTS = f"/opt/ml/checkpoints/{sm_args.spot_ckpt.split('/')[-1]}"
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_url", type=str, default='tcp://127.0.0.1:49152')
    parser.add_argument("--config_file", type=str, default="/opt/ml/code/detectron2/configs/quick_schedules/ped_mask_rcnn_R_50_FPN_training_acc_test.yaml") # 
    parser.add_argument("--eval_only", type=bool, default=False)
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--local_rank", default=0)
    parser.add_argument("--machine_rank", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--resume", type=str, default="False")
    parser.add_argument('--spot_ckpt', type=str, default=None)
    parser.add_argument("--opts", type=list, default=[])
    return parser.parse_args()



def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = parse_args()

    os.system('mkdir /opt/ml/checkpoints')
    os.system('mkdir /opt/ml/code/source')
    os.system('echo show opt/ml')
    os.system('ls /opt/ml/')
    os.system('echo ----------------')
    os.system('echo show config')
    os.system('ls /opt/ml/input/config')
    os.system('echo ----------------')
    os.system('ls /opt/ml/input')
    os.system('echo ----------------')
    os.system('ls /opt/ml/input/data/train')
    os.system('echo ----------------')
    os.system('ls /opt/ml/code')
    os.system('echo ----------------')
    os.system('ls /opt')
    os.system('echo ----------------')
    os.system('ls /')
    os.system('echo ----------------')
    os.system('ls /home')
    os.system('echo ----------------')

    os.system('cp /opt/ml/code/ped_mask_rcnn_R_50_FPN_training_acc_test.yaml /opt/ml/code/detectron2/configs/quick_schedules/ped_mask_rcnn_R_50_FPN_training_acc_test.yaml')
    
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
