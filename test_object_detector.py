#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:16:18 2021

@author: ombretta
"""

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

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
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
    parser.add_argument('--checkpoint_path',
            default = '',
            dest = 'checkpoint_path',
            type=str,
            help='Path to checkpoint we want to evaluate')
    parser.add_argument('--loss_function',
            default = 'smooth_l1',
            dest = 'loss_function',
            type=str,
            help='Which loss function we want to optimize (smooth_l1 or asymmetric_smooth_l1')

    args = parser.parse_args()
    return args

def main(dataDir='/Users/ombretta/Documents/Code/evaluating_IoU/coco', 
         cpu=True,
         model_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
         savedir='/Users/ombretta/Documents/Code/evaluating_IoU/coco',
         checkpoint_path=''):
    
    register_coco_instances("COCO_2017_train_ombretta", {}, os.path.join(dataDir, 
        "annotations/instances_train2017.json"), os.path.join(dataDir, "train2017/"))
    register_coco_instances("COCO_2017_val_ombretta", {}, os.path.join(dataDir, 
        "annotations/instances_val2017.json"), os.path.join(dataDir, "val2017/"))
    
    cfg = get_cfg()
    if cpu: cfg.MODEL.DEVICE='cpu'
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.DATASETS.TRAIN = ("COCO_2017_train_ombretta",)
    cfg.DATASETS.TEST = ("COCO_2017_val_ombretta",)

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = checkpoint_path  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    
    evaluator = COCOEvaluator("COCO_2017_val_ombretta", output_dir=savedir)
    val_loader = build_detection_test_loader(cfg, "COCO_2017_val_ombretta")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`

   
if __name__ == "__main__":
    args = get_args()
    all_IoUs, all_IoAs, all_areas_diff = main(args.dataDir, args.cpu, 
                                        args.model_file, args.savedir)