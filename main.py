#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:42:56 2021

@author: ombretta
"""


# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
#assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
from detectron2 import structures
setup_logger()

# import some common libraries
import numpy as np
import os, cv2
import pickle as pkl
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# for the COCO api
from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# visualize bounding boxes
from bounding_box import bounding_box as bb


def get_args(arguments_string=None, save=True):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', 
            default = './coco', 
            dest = 'dataDir',
            type=str,
            help='Annotations directory directory')
    parser.add_argument('--dataType', 
            default = 'val2017', 
            dest = 'dataType',
            type=str,
            help='Dataset type')
    parser.add_argument('--cpu', 
            default = False, 
            dest = 'cpu',
            action = 'store_true',
            help='Do you want to use cpu only?')
    parser.add_argument('--num_images', 
            default = -1, 
            dest = 'num_images',
            type=int,
            help='How many imagesdo you want to process? Default:all')
    parser.add_argument('--model_file', 
            default = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 
            dest = 'model_file',
            type=str,
            help='Config file for a detectron2 pretrained model')
    parser.add_argument('--savedir', 
            default = './coco', 
            dest = 'savedir',
            type=str,
            help='Where to save the results')
    parser.add_argument('--model_weights', 
            default = '', 
            dest = 'model_weights',
            type=str,
            help='Checkpoint to load. Default: detedtron2 trained model.')
    parser.add_argument('--fig_path', 
            default = "./Example "+\
                "predictions with smooth L1 loss/", 
            dest = 'fig_path',
            type=str,
            help='Where to save the visulizations.')
    parser.add_argument('--min_IoU', 
            default = 0.0, 
            dest = 'min_IoU',
            type=float,
            help='Min_IoU to consider')
    parser.add_argument('--max_IoU', 
            default = 1.0, 
            dest = 'max_IoU',
            type=float,
            help='Max_IoU to consider')
    parser.add_argument('--scaling_factor', 
            default = 1.0, 
            dest = 'scaling_factor',
            type=float,
            help='Scale bbox in the visualization by a certain factor')
    parser.add_argument('--save_single_boxes', 
            default = False, 
            dest = 'save_single_boxes',
            action = 'store_true',
            help='Do you want to save each predicted box in a single image?')
    args = parser.parse_args()
    return args

def show_and_save(title, image, path):
    cv2.imwrite(path, image)
    cv2.imshow(title, image)
    print("Press 'Enter' to display the next picture...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_image(window_name, image, text='', fig_path=''):
    plt.figure()
    plt.axis('off')
    plt.imshow(image[:, :, ::-1])
    if text != '':
        plt.text(20, 20, text, bbox=dict(fill=True, facecolor='red', 
                                         color='white', linewidth=2))
    if fig_path != '': 
        plt.savefig(fig_path)
    # plt.show() # Uncomment later

def read_image(impath, show=False):
    image = cv2.imread(impath)
    if show: 
        show_image(impath, image)
    return image

def detect_objects(image, predictor):
    outputs = predictor(image)
    return outputs, outputs["instances"].pred_classes, outputs["instances"].pred_boxes

def visualize_objects(image, outputs, configs, fig_path=''):
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(configs.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    show_image('', out.get_image(), fig_path=fig_path)

def get_gt_bboxes_coco_api(image_annotations, convert_coco_ids):
    num_non_crowd_bboxes = len(image_annotations)
    gt_boxes = np.zeros([num_non_crowd_bboxes, 4])
    gt_classes = np.zeros([num_non_crowd_bboxes], dtype=int)
    object_areas = np.zeros([num_non_crowd_bboxes])
    for i, box in enumerate(image_annotations):
        # print("image_annotations[i]['bbox']", image_annotations[i]['bbox'])
        x1 = image_annotations[i]['bbox'][0]
        y1 = image_annotations[i]['bbox'][1]
        w = image_annotations[i]['bbox'][2]
        h = image_annotations[i]['bbox'][3]
        x2, y2 = x1+w, y1+h
        gt_boxes[i] = np.array([x1, y1, x2, y2])
        gt_classes[i] = int(convert_coco_ids[image_annotations[i]['category_id']])
        object_areas[i] = image_annotations[i]['area']
    return gt_boxes, gt_classes, object_areas

def visualize_bboxes(image, bboxes_gt, gt_classes, bboxes_pred, 
                         classes_pred, class_catalog):
    for i, gt_box in enumerate(bboxes_gt):
        left, bottom = gt_box[0].item(), gt_box[1].item()
        right, top = gt_box[2].item(), gt_box[3].item()
        gt_class = int(gt_classes[i])
        label = class_catalog[gt_class] if gt_class<len(class_catalog) else ''
        bb.add(image, left, top, right, bottom) #, label)
    show_and_save('test.jpg', image, 'test.jpg')

def print_bb_area(bb_type, bboxes):
    for bbox in bboxes:
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        print("Area", bb_type, area)
   
def print_bboxes(bboxes, classes, class_catalog):
    for idx, coordinates in enumerate(bboxes):
            class_index = classes[idx]
            class_name = class_catalog[class_index]
            print(class_name, coordinates)
                  
def save_data(data, filename, with_torch=False):
    with open(filename, "wb") as f:
        if with_torch == True: torch.save(data, f)
        else: pkl.dump(data, f)

def load_data(filename, with_torch = False):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = torch.load(f, map_location='cpu') if with_torch == True else pkl.load(f)
        return data
    else: print("File", filename, "does not exists.")    
    
                  
def main(dataDir='./coco', 
         dataType='val2017', cpu=True, num_images=-1, 
         model_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
         savedir='./coco',
         model_weights=""):
    
    #Initialize detectron 
    cfg = get_cfg()
    # on cpu
    if cpu: cfg.MODEL.DEVICE='cpu'
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(model_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    if model_weights == "":
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_file)
    else:
        cfg.MODEL.WEIGHTS = model_weights
    predictor = DefaultPredictor(cfg)
    
    # load coco
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_catalog = metadata.thing_classes
    
    # use COCO API for the categories
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    # initialize COCO api for instance annotations
    coco=COCO(annFile)
    # display COCO categories and supercategories
    cat_names = [cat['name'] for cat in coco.loadCats(coco.getCatIds())]
    catIds = coco.getCatIds();
    convert_coco_ids = {}
    for i in range(len(catIds)):
        convert_coco_ids[catIds[i]] = i
        
    # track IoUs, IoAs, areas per bounding box
    all_IoUs, all_IoAs, all_areas_diff, all_object_areas = [], [], [], []
    all_gt_areas, all_pred_areas = [], []

    # loop over coco validation set
    for ids, im_name in enumerate(os.listdir(os.path.join(dataDir, 'images'))[:10]):#[:num_images]):
        
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
            # plt.imshow(I); plt.axis('off')
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            # coco.showAnns(anns, True)
            # plt.show()
            
            # predict bboxes
            outputs, classes_pred, bboxes_pred = detect_objects(I, predictor)
            # visualize_objects(I, outputs, cfg)
            
            # evaluate
            n_gt_objects = len(anns)
            n_pred_objects = len(bboxes_pred)
            print('Predicted', n_pred_objects, "out of", n_gt_objects)
            
            bboxes_gt, gt_classes, object_areas_gt = get_gt_bboxes_coco_api(
                anns, convert_coco_ids)
            bboxes_gt = structures.Boxes(torch.Tensor(bboxes_gt))
            # visualize_bboxes(I, bboxes_gt, gt_classes, bboxes_pred, 
            #                   classes_pred, cat_names)
            
            # Intersection over area and Intersection over Union. 
            # Output: predictions x ground truth
            IOUs = structures.pairwise_iou(bboxes_pred.to('cpu'), bboxes_gt)
            IOAs = structures.pairwise_ioa(bboxes_pred.to('cpu'), bboxes_gt)
            # print("IoUs", IOUs)
            # print("IoAs", IOAs)
            # print("Area gt", bboxes_gt.area())
            # print("Area pred", bboxes_pred.area())
            
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
                print(best_bb)
                print("best_IoU", best_IoU)
                print("best_IoA", best_IoA)
                diff_area_gt_area_pred = torch.zeros(n_gt_objects)
                object_areas = torch.zeros(n_gt_objects)
                for i in range(n_gt_objects): 
                    print(i, "diff_area_gt_area_pred[i]", diff_area_gt_area_pred[i],  "object_areas_gt[i] ", object_areas_gt[i] )
                    diff_area_gt_area_pred[i] = \
                        bboxes_gt.area()[i]-bboxes_pred.area()[best_bb[i]]
                    object_areas[i] = object_areas_gt[i] 
                print("area_gt - area_pred", diff_area_gt_area_pred)
                gt_areas = bboxes_gt.area()
                pred_areas = bboxes_pred.area()

            else: 
                best_IoU = torch.tensor([float('nan')])
                best_bb = torch.tensor(float('nan'))
                best_IoA = torch.tensor([float('nan')])
                diff_area_gt_area_pred = torch.tensor([float('nan')])
                object_areas = torch.tensor([float('nan')])
                gt_areas = torch.tensor([float('nan')])
                pred_areas = torch.tensor([float('nan')])
            
            if len(all_IoUs) == 0: 
                all_IoUs = best_IoU
                all_IoAs = best_IoA
                all_areas_diff = diff_area_gt_area_pred
                all_object_areas = object_areas
                all_gt_areas = gt_areas
                all_pred_areas = pred_areas
            else: 
                all_IoUs = torch.cat((all_IoUs, best_IoU)) 
                all_IoAs = torch.cat((all_IoAs, best_IoA)) 
                all_areas_diff = torch.cat((all_areas_diff, diff_area_gt_area_pred)) 
                all_object_areas = torch.cat((all_object_areas, object_areas))
                all_gt_areas = torch.cat((all_gt_areas, gt_areas))
                all_pred_areas = torch.cat((all_pred_areas, pred_areas))

            print("len(all_object_areas)==len(all_areas_diff)", len(all_object_areas)==len(all_areas_diff), len(all_object_areas), len(all_areas_diff))
            # print("gt classes")
            # print_bboxes(bboxes_gt, gt_classes, class_catalog)
                
            # print("predicted classes")
            # print_bboxes(bboxes_pred, classes_pred, class_catalog)
    
    save_data(all_IoUs, os.path.join(savedir, "val2017_IoUs.dat"))
    save_data(all_IoAs, os.path.join(savedir, "val2017_IoAs.dat"))
    save_data(all_areas_diff, os.path.join(savedir, "val2017_areas_diff.dat"))
    save_data(all_object_areas, os.path.join(savedir, "val2017_object_areas.dat"))
    save_data(all_gt_areas, os.path.join(savedir, "val2017_gt_areas.dat"))
    save_data(all_pred_areas, os.path.join(savedir, "val2017_pred_areas.dat"))

        
    return all_IoUs, all_IoAs, all_areas_diff
        
#%%
if __name__ == "__main__":
    args = get_args()
    all_IoUs, all_IoAs, all_areas_diff = main(args.dataDir, args.dataType, 
                                              args.cpu, args.num_images,
                                              args.model_file, args.savedir,
                                              args.model_weights)
