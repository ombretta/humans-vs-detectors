#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:42:56 2021

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

from main import get_args, show_image, detect_objects, get_gt_bboxes_coco_api,\
    visualize_objects


def visualize_bboxes(image, bboxes, classes, class_catalog, colors='', 
                     text='', fig_path=''):
    if colors == '': colors = ['red' for i in range(len(bboxes))]
    for i, gt_box in enumerate(bboxes):
        left, bottom = gt_box[0].item(), gt_box[1].item()
        right, top = gt_box[2].item(), gt_box[3].item()
        gt_class = int(classes[i])
        label = class_catalog[gt_class] if gt_class<len(class_catalog) else ''
        bb.add(image, left, top, right, bottom, label, color=colors[i])
    show_image('prova.jpg', image[:, :, ::-1], text, fig_path)
    
                  
def main(dataDir='./coco',
         dataType='val2017', cpu=True, num_images=-1, 
         model_file="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
         model_weights='',  
         fig_path = ""):
    
    #Initialize detectron 
    cfg = get_cfg()
    # on cpu
    if cpu: cfg.MODEL.DEVICE='cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    
    if model_weights == "":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
        fig_path = "example_prediciton_detectron2_pretrained_model"
    else:
        print("Loading weights", model_weights)
        cfg.MODEL.WEIGHTS = model_weights
        model_name = "".join(model_weights.split("/")[-1].split(".")[:-1])
        fig_path = "example_predictions_with_" + model_name
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
    
    num_figs = 4
    count_IoU_areas = {}
    incomplete = 7
    
    # loop over coco validation set
    for ids, im_name in enumerate(os.listdir(os.path.join(dataDir, 'images'))):#[:num_images]):
        
        if incomplete <= 0: break 
    
        # load image
        impath = os.path.join(dataDir, 'images', im_name)
        
        print("\n", ids, impath)
        im_id = int(im_name.split('.')[0])
        img = coco.loadImgs(im_id)[0]
        I = io.imread(img['coco_url'])
        # skip black and white images
        if len(I.shape) > 2:
            show_image('', I[:, :, ::-1])
            
            # load and display instance annotations
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            
            # predict bboxes
            outputs, classes_pred, bboxes_pred = detect_objects(I, predictor)
            save_im_path = fig_path+str(im_id)+"_all_predictions.png"
            visualize_objects(I, outputs, cfg, fig_path=save_im_path)
            
            # evaluate
            n_gt_objects = len(anns)
            n_pred_objects = len(bboxes_pred)
            print('Predicted', n_pred_objects, "out of", n_gt_objects)
            
            bboxes_gt, gt_classes, object_areas = get_gt_bboxes_coco_api(anns, convert_coco_ids)
            bboxes_gt = structures.Boxes(torch.Tensor(bboxes_gt))
            
            save_im_path = fig_path+str(im_id)+"_all_gt.png"
            visualize_bboxes(I.copy(), bboxes_gt, gt_classes, class_catalog,
                             fig_path=save_im_path)
            
            IOUs = structures.pairwise_iou(bboxes_pred, bboxes_gt)
            IOAs = structures.pairwise_ioa(bboxes_pred, bboxes_gt)
            
            # Keep only boxes with the correctly predict class
            wrong_class = torch.ones([n_pred_objects, n_gt_objects], dtype=bool)
            for i in range(n_pred_objects):
                for j in range(n_gt_objects):
                    if classes_pred[i] == gt_classes[j]: wrong_class[i,j] = False
            IOUs[wrong_class] = 0
            
            if IOUs.shape[0]>0: 
                best_IoU, best_bb = torch.max(IOUs, axis=0)
                best_IoA = torch.zeros(n_gt_objects)
                for i in range(n_gt_objects): best_IoA[i] = IOAs[best_bb[i], i]
                diff_area_gt_area_pred = torch.zeros(n_gt_objects)
                for i in range(n_gt_objects): 
                    diff_area_gt_area_pred[i] = \
                        bboxes_gt.area()[i]-bboxes_pred.area()[best_bb[i]]
                    
                for i in range(n_gt_objects): 
                    
                    best_IoU_i = math.floor(float(best_IoU[i])*10)/10
                    if best_IoU_i not in count_IoU_areas: 
                        count_IoU_areas[best_IoU_i] = {}
                        count_IoU_areas[best_IoU_i]['small'] = 0
                        count_IoU_areas[best_IoU_i]['large'] = 0
                        
                    figpath = fig_path + str(im_id)+"_IoU"+\
                        str(best_IoU_i)
                
                    if diff_area_gt_area_pred[i] > 0:
                        size = 'small'
                    else: 
                        size = 'large'
                    figpath+="_"+size+".jpg"
                    
                    if count_IoU_areas[best_IoU_i][size] < num_figs: 
                        count_IoU_areas[best_IoU_i][size] += 1
                
                        if count_IoU_areas[best_IoU_i]['small'] == num_figs \
                            and count_IoU_areas[best_IoU_i]['large'] == num_figs:
                                incomplete -= 1
                                print("incomplete", incomplete)
                                
                        if os.path.exists(figpath): break
                    
                        if gt_classes[i] == classes_pred[best_bb[i]]:
                            gt = bboxes_gt[i]
                            pred = bboxes_pred[int(best_bb[i])]
                            bboxes = structures.Boxes.cat((gt, pred))
                            classes = [gt_classes[i]] + [classes_pred[best_bb[i]]]
                            colors = ['red', 'blue']
                            text = 'IoU = '+str(best_IoU_i)
                            visualize_bboxes(I.copy(), bboxes, classes, 
                                             class_catalog, colors, text, figpath) 
                print(incomplete, count_IoU_areas)
    
        
#%%
if __name__ == "__main__":
    args = get_args()
    main(args.dataDir, args.dataType, True, args.num_images, args.model_file, args.model_weights, args.fig_path)
