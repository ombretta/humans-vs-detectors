#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 13:44:50 2021

@author: ombretta
"""

import os
from Analyze_IoUs import load_data
import matplotlib.pyplot as plt
import numpy as np


def autolabel(ax, rects, float_values = False):
    for rect in rects:
        h = rect.get_height()
        if not float_values:
            ax.text(rect.get_x()+rect.get_width()/2., 1.0+h, '%d'%int(h),
                ha='center', va='bottom')
        else:
            ax.text(rect.get_x()+rect.get_width()/2., 1.0+h, '%.2f'%float(h),
                ha='center', va='bottom')

model_name = "faster_rcnn_R_50_FPN_3x"
# model_name = "retinanet_R_50_FPN_1x"
# model_name = "cascade_mask_rcnn_R_50_FPN_3x"

suffix1 = "_bboxes_analysis"
# suffix1 = "_detectron2_pretrained_asymmetric_smooth_l1_factor2_beta1.0_lr0.00025_bboxes_analysis_70k_iters"
suffix2 = "_detectron2_pretrained_asymmetric_smooth_l1_factor2_beta1.0_lr0.00025_bboxes_analysis_70k_iters"

bars_names = ('Smooth L1 loss - small boxes', 'Smooth L1 loss - large boxes', 
                'Asymmetric smooth L1 loss - small boxes', 
                'Asymmetric smooth L1 loss - large boxes')

# bars_names = ('Asymmetric smooth L1 loss, factor 2, beta=1, (70k iters) - small boxes', 
#               'Asymmetric smooth L1 loss, factor 2 beta=1, (70k iters) - large boxes', 
#               'Asymmetric smooth L1 loss, factor 2 (70k iters) - small boxes', 
#               'Asymmetric smooth L1 loss, factor 2 (70k iters) - large boxes')

models = [model_name+suffix1, 
          model_name+suffix2]

data_dir_root = "./IoUs_IoAs_Areas/"
all_counts, S_counts, M_counts, L_counts = {}, {}, {}, {}

for model in models:
    print("model", model, len(model))
    dataDir = os.path.join(data_dir_root, model)
    print(dataDir)
    all_counts[model] = load_data(os.path.join(dataDir, "bboxes_count.dat"))
    S_counts[model] = load_data(os.path.join(dataDir, "bboxes_count_small_objects.dat"))
    M_counts[model] = load_data(os.path.join(dataDir, "bboxes_count_medium_objects.dat"))
    L_counts[model] = load_data(os.path.join(dataDir, "bboxes_count_large_objects.dat"))

x = np.array([round(i*0.1,1) for i in range(9,2, -1)])

for obj, counts in zip(["All", "Small", "Medium", "Large"],
                       [all_counts, S_counts, M_counts, L_counts]):
    
    fig = plt.figure(figsize=[18,10])
    ax = fig.add_subplot(111)
    ax.title.set_text("Predictions with "+" ".join(model_name.split("_"))+" - "+obj+" objects")
    
    rects1_m1 = ax.bar(x-0.04, counts[model_name+suffix1]['small'], width=0.01, color='b', align='center')
    rects2_m1 = ax.bar(x-0.03, counts[model_name+suffix1]['large'], width=0.01, color='r', align='center')
    for i, b in enumerate(rects1_m1):
        b.set_color(plt.cm.Blues(70))
    for i, b in enumerate(rects2_m1):
        b.set_color(plt.cm.Blues(110))
    
    rects1_m2 = ax.bar(x-0.01, counts[model_name+suffix2]['small'], width=0.01, color='b', align='center')
    rects2_m2 = ax.bar(x-0.00, counts[model_name+suffix2]['large'], width=0.01, color='r', align='center')
    for i, b in enumerate(rects1_m2):
        b.set_color(plt.cm.Greens(70))
    for i, b in enumerate(rects2_m2):
        b.set_color(plt.cm.Greens(110))
    
    y_lim = max(max(counts[model_name+suffix1]['small']),
                max(counts[model_name+suffix1]['large']),
                max(counts[model_name+suffix2]['small']),
                max(counts[model_name+suffix2]['large'])) + 200
    plt.ylim(top=y_lim)

    ax.set_ylabel('Bboxes count')
    ax.set_xlabel('IoU range')
    ax.set_xticklabels( ('', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', \
                          '0.8-0.9', '0.9-1.0') )
    
    ax.legend( (rects1_m1[0], rects2_m1[0], rects1_m2[0], rects2_m2[0]), 
               bars_names, 
               prop={'size': 16})
    
    autolabel(ax, rects1_m1)
    autolabel(ax, rects2_m1)
    autolabel(ax, rects1_m2)
    autolabel(ax, rects2_m2)
    
    plt.savefig(os.path.join(data_dir_root, model_name+"_"+suffix1+"_"+suffix2+"_"+obj+"_objects_losses_comparison.png"))