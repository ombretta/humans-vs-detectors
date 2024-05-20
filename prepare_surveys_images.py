#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:49:06 2021

@author: ombretta
"""

# check pytorch installation: 
import torch
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
from detectron2 import structures
setup_logger()

# import some common libraries
import os
import math
import numpy as np

import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

# for the COCO api
from pycocotools.coco import COCO
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# visualize bounding boxes
from bounding_box import bounding_box as bb
from store_surveys_images import get_obj_category
from main import get_args, show_image, detect_objects, get_gt_bboxes_coco_api,\
    visualize_objects


def crop_image(image, bboxes, fig_path=''):
    for i, gt_box in enumerate(bboxes):
        left, bottom = gt_box[0].item(), gt_box[1].item()
        right, top = gt_box[2].item(), gt_box[3].item()
    cropped_image = image[left:right, bottom:top, ::-1]
    plt.figure()
    plt.axis('off')
    plt.imshow(cropped_image[:, :, ::-1])
    plt.savefig(fig_path)

def visualize_bboxes(image, bboxes, classes, class_catalog=None, colors='',
                     text='', fig_path=''):
    if colors == '': colors = ['green' for i in range(len(bboxes))]
    for i, gt_box in enumerate(bboxes):
        left, bottom = gt_box[0].item(), gt_box[1].item()
        right, top = gt_box[2].item(), gt_box[3].item()
        gt_class = int(classes[i])
        if class_catalog != None:
            label = class_catalog[gt_class] if gt_class<len(class_catalog) else ''
        else: label = None
        bb.add(image, left, top, right, bottom, label, color=colors[i])
    show_image('prova.jpg', image[:, :, ::-1], text, fig_path)
    
def rescale_boxes(bboxes_pred, scaling_factor, height, width):
    scaled_bboxes = []
    for idx, box in enumerate(bboxes_pred): 
        left, bottom = box[0].item(), box[1].item()
        right, top = box[2].item(), box[3].item()
        y_shift = ((top-bottom)*scaling_factor-(top-bottom))/2
        x_shift = ((right-left)*scaling_factor-(right-left))/2
        left -= x_shift
        bottom -= y_shift 
        right += x_shift
        top += y_shift        
        
        scaled_box = torch.Tensor([[max(0,left), max(0,bottom), 
                                    min(width,right), min(height,top)]])
        if len(scaled_bboxes)==0: scaled_bboxes = scaled_box
        else: scaled_bboxes = torch.cat((scaled_bboxes, scaled_box)) 
    return structures.Boxes(scaled_bboxes)
    
    
                  
def main(dataDir='./coco',
         dataType='val2017', cpu=True, num_images=1, 
         model_file="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
         model_weights='', min_IoU=0.0, max_IoU=1.0, scaling_factor=1.0,
         save_single_boxes=False,
         fig_path="./surveys_images/"):
    
    #Initialize detectron 
    cfg = get_cfg()
    # on cpu
    if cpu: cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    
    if "retinanet_R_50_FPN_1x" in model_weights: 
        model_file = "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    if "cascade_mask_rcnn_R_50_FPN_3x" in model_weights: 
        model_file = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    model_name = model_file.split("/")[-1].split(".")[0]
    
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    
    if model_weights == "":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
        fig_path += model_name+"detectron2_pretrained"
    else:
        print("Loading weights", model_weights)
        cfg.MODEL.WEIGHTS = model_weights
        model_name = "".join("_".join(model_weights.split("/")[-2:]).split(".")[:-1])
        fig_path += model_name
        
    fig_path += "_scaling"+str(scaling_factor)+"/" if scaling_factor != 1.0 else "/"
    if not os.path.exists(fig_path): os.mkdir(fig_path) 
        
    fig_path += str(min_IoU)+"<IoU<"+str(max_IoU)+"/"
        
    if not os.path.exists(fig_path): os.mkdir(fig_path) 
    
    predictor = DefaultPredictor(cfg)
    
    # load coco
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_catalog = metadata.thing_classes
    
    # use COCO API for the categories
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
    # display COCO categories and supercategories
    catIds = coco.getCatIds();
    convert_coco_ids = {}
    for i in range(len(catIds)):
        convert_coco_ids[catIds[i]] = i
    
    # loop over coco validation set
    for ids, im_name in enumerate(os.listdir(os.path.join(dataDir, 'images'))[:num_images]):
        
        # load image
        impath = os.path.join(dataDir, 'images', im_name)
        
        print("\n", ids, impath)
        im_id = int(im_name.split('.')[0])
        img = coco.loadImgs(im_id)[0]
        I = io.imread(img['coco_url'])
        # skip black and white images
        if len(I.shape) > 2:
            # show_image('', I[:, :, ::-1])
            
            # load and display instance annotations
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            
            # predict bboxes
            outputs, classes_pred, bboxes_pred = detect_objects(I, predictor)
            # visualize_objects(I, outputs, cfg, fig_path=save_im_path)
            
            bboxes_pred = rescale_boxes(bboxes_pred, scaling_factor,
                                        I.shape[0], I.shape[1])
            
            visualize_bboxes(I.copy(), bboxes_pred, classes_pred, class_catalog,
                             fig_path=fig_path+str(im_id)+"_all_predictions.png")
            
            # evaluate
            n_gt_objects = len(anns)
            n_pred_objects = len(bboxes_pred)
            print('Predicted', n_pred_objects, "out of", n_gt_objects)
            
            bboxes_gt, gt_classes, object_areas = get_gt_bboxes_coco_api(anns, convert_coco_ids)
            bboxes_gt = structures.Boxes(torch.Tensor(bboxes_gt))
            
            visualize_bboxes(I.copy(), bboxes_gt, gt_classes, class_catalog,
                              fig_path=fig_path+str(im_id)+"_all_gt.png")
            
            IOUs = structures.pairwise_iou(bboxes_pred, bboxes_gt)
            IOAs = structures.pairwise_ioa(bboxes_pred, bboxes_gt)
            
            # Keep only boxes with the correctly predict class
            wrong_class = torch.ones([n_pred_objects, n_gt_objects], dtype=bool)
            # areas = torch.zeros([n_pred_objects], dtype=float)
            areas = torch.zeros([n_pred_objects, n_gt_objects], dtype=float)
            for i in range(n_pred_objects):
                for j in range(n_gt_objects):
                    if classes_pred[i] == gt_classes[j]:
                        wrong_class[i,j] = False
                        areas[i,j] = object_areas[j].item()
            IOUs[wrong_class] = 0
            
            # filter accroding to IoU
            best_IoU, best_bb = torch.max(IOUs, axis=1)
            best_areas = areas[torch.arange(n_pred_objects), best_bb]
            IoU_selected = np.logical_and(best_IoU>min_IoU, best_IoU<=max_IoU)
                    
            sorted_classes_pred = list(set(sorted([int(c) for c in classes_pred])))    
            for obj_class in sorted_classes_pred:
                class_name = class_catalog[obj_class]
                print("obj_class", class_name)
                selected_rows = np.logical_and(classes_pred==obj_class, IoU_selected)
                if any(selected_rows):
                    fig_name = fig_path+str(im_id)+"_"+class_name
                    # visualize_bboxes(I.copy(), bboxes_pred[selected_rows],
                    #         classes_pred[selected_rows], class_catalog,
                    #         fig_path=fig_name+".png")
                    
                    # Save the single boxes in separate images 
                    if save_single_boxes and len(bboxes_pred[selected_rows])>=1:
                        for box_num, (box, class_pred, iou, area) in enumerate(zip(
                            bboxes_pred[selected_rows], classes_pred[selected_rows],
                            best_IoU[selected_rows], best_areas[selected_rows])):
                            obj_cat = get_obj_category(area)
                            box_name = fig_name + str(box_num) + "_" + obj_cat + "_IoU" + \
                            str(np.floor(iou * 10) / 10) + ".png"
                            visualize_bboxes(I.copy(), [box], [class_pred], class_catalog,
                                fig_path=box_name)
                            crop_image(I.copy(), [box], fig_path=box_name.replace(".png", "_crop.png"))
                    

if __name__ == "__main__":
    args = get_args()
    main(args.dataDir, args.dataType, True, args.num_images,
        args.model_file, args.model_weights, args.min_IoU, args.max_IoU,
        args.scaling_factor, args.save_single_boxes, args.fig_path)
