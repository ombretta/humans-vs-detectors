#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:34:33 2021

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


models = ["faster_rcnn_R_50_FPN_3x", "retinanet_R_50_FPN_1x", "cascade_mask_rcnn_R_50_FPN_3x"]

data_dir_root = "./IoUs_IoAs_Areas/"
all_counts, S_counts, M_counts, L_counts = {}, {}, {}, {}

for model in models:
    print("model", model, len(model))
    dataDir = os.path.join(data_dir_root, model+"_bboxes_analysis")
    print(dataDir)
    all_counts[model] = load_data(os.path.join(dataDir, "bboxes_count.dat"))
    S_counts[model] = load_data(os.path.join(dataDir, "bboxes_count_small_objects.dat"))
    M_counts[model] = load_data(os.path.join(dataDir, "bboxes_count_medium_objects.dat"))
    L_counts[model] = load_data(os.path.join(dataDir, "bboxes_count_large_objects.dat"))

x = np.array([round(i*0.1,1) for i in range(9,2, -1)])

fig = plt.figure(figsize=[18,10])
ax = fig.add_subplot(111)
plt.title("Comparison of object detectors predictions")

rects1_m1 = ax.bar(x-0.04, all_counts['faster_rcnn_R_50_FPN_3x']['small'], width=0.01, color='b', align='center')
rects2_m1 = ax.bar(x-0.03, all_counts['faster_rcnn_R_50_FPN_3x']['large'], width=0.01, color='r', align='center')
for i, b in enumerate(rects1_m1):
    b.set_color(plt.cm.Blues(70))
for i, b in enumerate(rects2_m1):
    b.set_color(plt.cm.Blues(110))

rects1_m2 = ax.bar(x-0.01, all_counts['retinanet_R_50_FPN_1x']['small'], width=0.01, color='b', align='center')
rects2_m2 = ax.bar(x-0.00, all_counts['retinanet_R_50_FPN_1x']['large'], width=0.01, color='r', align='center')
for i, b in enumerate(rects1_m2):
    b.set_color(plt.cm.Greens(70))
for i, b in enumerate(rects2_m2):
    b.set_color(plt.cm.Greens(110))

rects1_m3 = ax.bar(x+0.02, all_counts['cascade_mask_rcnn_R_50_FPN_3x']['small'], width=0.01, color='b', align='center')
rects2_m3 = ax.bar(x+0.03, all_counts['cascade_mask_rcnn_R_50_FPN_3x']['large'], width=0.01, color='r', align='center')
for i, b in enumerate(rects1_m3):
    b.set_color(plt.cm.Reds(70))
for i, b in enumerate(rects2_m3):
    b.set_color(plt.cm.Reds(110))


plt.ylim(top=5000)
ax.set_ylabel('Bboxes count')
ax.set_xlabel('IoU range')
ax.set_xticklabels( ('', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', \
                      '0.8-0.9', '0.9-1.0') )
ax.legend( (rects1_m1[0], rects2_m1[0], rects1_m2[0], rects2_m2[0], rects1_m3[0], rects2_m3[0]), 
          ('Faster rcnn R50 FPN 3x: small boxes', 'Faster rcnn R50 FPN 3x: large boxes', 
           'Retinanet R50 FPN 1x: small boxes', 'Retinanet R50 FPN 1x: large boxes', 
           'Cascade mask rcnn R50 FPN 3x: small boxes', 'Cascade mask rcnn R50 FPN 3x: large boxes'), 
           prop={'size': 16})
autolabel(ax, rects1_m1)
autolabel(ax, rects2_m1)
autolabel(ax, rects1_m2)
autolabel(ax, rects2_m2)
autolabel(ax, rects1_m3)
autolabel(ax, rects2_m3)
plt.savefig(os.path.join(data_dir_root, "models_IoUs_comparison.png"))