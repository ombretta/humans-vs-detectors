#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:01:26 2021

@author: ombretta
"""


''' implement model training from https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=7unkuuiqLdqd '''

# check pytorch installation: 
import torch
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances


def get_args(arguments_string=None, save=True):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', 
            default = '/Users/ombretta/Documents/Code/evaluating_IoU/coco', 
            dest = 'dataDir',
            type=str,
            help='Annotations directory directory')
    parser.add_argument('--cpu', 
            default = False, 
            dest = 'cpu',
            action = 'store_true',
            help='Do you want to use cpu only?')
    parser.add_argument('--model_file', 
            default = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
            dest = 'model_file',
            type=str,
            help='Config file for a detectron2 pretrained model')
    parser.add_argument('--savedir', 
            default = '/Users/ombretta/Documents/Code/evaluating_IoU/coco', 
            dest = 'savedir',
            type=str,
            help='Where to save the results')
    args = parser.parse_args()
    return args


def main(dataDir='/Users/ombretta/Documents/Code/evaluating_IoU/coco', 
         cpu=True,
         model_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
         savedir='/Users/ombretta/Documents/Code/evaluating_IoU/coco'):
    
    register_coco_instances("COCO_dataset_train", {}, os.path.join(dataDir, 
        "annotations/instances_train2017.json"), os.path.join(dataDir, "train2017/"))
    register_coco_instances("COCO_dataset_val", {}, os.path.join(dataDir, 
        "annotations/instances_val2017.json"), os.path.join(dataDir, "val2017/"))
    
    cfg = get_cfg()
    if cpu: cfg.MODEL.DEVICE='cpu'
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ("my_dataset_train")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = savedir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
if __name__ == "__main__":
    args = get_args()
    main(args.dataDir, args.cpu, args.model_file, args.savedir)