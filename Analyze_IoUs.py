#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 07:59:17 2021

@author: ombretta
"""

import os
import pickle as pkl
import torch
import numpy as np
import math

import matplotlib.pyplot as plt

from statsmodels.stats import weightstats as stests

#%%

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

def analyze_IoU_IoA_areas(IoUs, IoAs, area_diff, min_IoU = 0.0, max_IoU = 1.0):
    print(min_IoU, "< IoU <=", max_IoU)
    print(len(IoUs), "boxes")
    mean_IoU, std_IoU = np.nanmean(IoUs), np.nanstd(IoUs)
    mean_IoA, std_IoA = np.nanmean(IoAs), np.nanstd(IoAs)
    mean_area_diff, std_area_diff = np.nanmean(area_diff), np.nanstd(area_diff)
    print("Mean and std IoU", round(mean_IoU,3), round(std_IoU,3))
    print("Mean and std IoA", round(mean_IoA,3), round(std_IoA,3))
    print("Mean and std area diff", round(mean_area_diff,3), round(std_area_diff,3))
    small_area_count = len(area_diff[area_diff>0])
    large_area_count = len(area_diff[area_diff<0])
    print("Small bb", small_area_count, "- Large bb", large_area_count)
    z_test(small_area_count, large_area_count, "Z-test")
    normal_small_area_count = round(small_area_count/len(IoUs), 2)
    normal_large_area_count = round(large_area_count/len(IoUs), 2)
    print("\n")
    return small_area_count, large_area_count, normal_small_area_count, \
        normal_large_area_count

def count_boxes(IoUs, IoAs, areas_diff, s_count, l_count,
                s_count_norm, l_count_norm, min_IoU=0.0, max_IoU=1.0):
    IoU_selected = np.logical_and(IoUs>min_IoU, IoUs<=max_IoU)
    small, large, norm_small, norm_large = analyze_IoU_IoA_areas(
                    IoUs[IoU_selected],
                    IoAs[IoU_selected], areas_diff[IoU_selected],
                    min_IoU = min_IoU, max_IoU = max_IoU)
    s_count += [small]; l_count += [large]
    s_count_norm += [norm_small]; l_count_norm += [norm_large]
    # z_test(small, large, "Z-test")
    return s_count, l_count, s_count_norm, l_count_norm

def count_all_boxes(all_IoUs, all_IoAs, all_areas_diff, S_IoUs, S_IoAs,
                    S_areas_diff, M_IoUs, M_IoAs, M_areas_diff, L_IoUs,
                    L_IoAs, L_areas_diff, all_counts, S_counts, M_counts,
                    L_counts, min_IoU=0.0, max_IoU=1.0):
    # All
    all_counts['small'], all_counts['large'], all_counts['small_norm'], \
        all_counts['large_norm'] = count_boxes(all_IoUs, all_IoAs,
        all_areas_diff, all_counts['small'], all_counts['large'],
        all_counts['small_norm'], all_counts['large_norm'],
        min_IoU=min_IoU, max_IoU=max_IoU)
    # Small objects
    print("small")
    S_counts['small'], S_counts['large'], S_counts['small_norm'], \
        S_counts['large_norm'] = count_boxes(S_IoUs, S_IoAs,
        S_areas_diff, S_counts['small'], S_counts['large'],
        S_counts['small_norm'], S_counts['large_norm'],
        min_IoU=min_IoU, max_IoU=max_IoU)
    # Medium objects
    print("medium")
    M_counts['small'], M_counts['large'], M_counts['small_norm'], \
        M_counts['large_norm'] = count_boxes(M_IoUs, M_IoAs,
        M_areas_diff, M_counts['small'], M_counts['large'],
        M_counts['small_norm'], M_counts['large_norm'],
        min_IoU=min_IoU, max_IoU=max_IoU)
    # Large objects
    print("large")
    L_counts['small'], L_counts['large'], L_counts['small_norm'], \
        L_counts['large_norm'] = count_boxes(L_IoUs, L_IoAs,
        L_areas_diff, L_counts['small'], L_counts['large'],
        L_counts['small_norm'], L_counts['large_norm'],
        min_IoU=min_IoU, max_IoU=max_IoU)

    return all_counts, S_counts, M_counts, L_counts

def autolabel(ax, rects, float_values = False):
    for rect in rects:
        h = rect.get_height()
        if not float_values:
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
        else:
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%.2f'%float(h),
                ha='center', va='bottom')

def draw_histogram(x, small_count, large_count, figure_path, x_lim,
                   float_values=False, figzise=(8, 6)):
    fig = plt.figure(figsize=figzise)
    plt.style.use('ggplot')

    ax = fig.add_subplot(111)
    # rects1 = ax.bar(x-0.04, small_count, width=0.04, color="#f00080", align='center')
    # rects2 = ax.bar(x, large_count, width=0.04, color="#5b8dff", align='center')

    rects1 = ax.barh(x-0.04, small_count, 0.04, label='Small box', color="#f00080", align='center')
    rects2 = ax.barh(x, large_count, 0.04, label='Large box', color="#5b8dff", align='center')

    plt.xlim(right=x_lim)
    ax.set_xlabel('# boxes', fontsize=22)
    ax.set_ylabel('IoU Range', fontsize=22)

    # ax.legend( (rects1[0], rects2[0]), ('Small boxes', 'Large boxes'), fontsize=22)
    # autolabel(ax, rects1, float_values)
    # autolabel(ax, rects2, float_values)


    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    iou_labels = ('', r'$[0.3, 0.4)$', r'$[0.4, 0.5)$', r'$[0.5, 0.6)$',
          r'$[0.6, 0.7)$', r'$[0.7, 0.8)$', r'$[0.8, 0.9)$', r'$[0.9, 1.0)$')

    ax.set_yticklabels(iou_labels, fontsize=18)

    ax.legend(loc='upper center', bbox_to_anchor=(0.42, 1.15),
              ncol=2, fontsize=20)

    # plt.savefig(figure_path, bbox_inches='tight')

    fig.tight_layout()
    plt.savefig(figure_path, bbox_inches='tight', format='pdf', dpi=1200)

def z_test(count_small, count_large, title):
    print(title)
    preds = np.concatenate((np.ones([count_small]), np.zeros([count_large])))
    ztest, pval = stests.ztest(preds, value=0.5)
    print(round(float(pval), 3))
    if pval<0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")

#%%
def main(dataDir = ""):

    all_IoUs = np.array(load_data(os.path.join(dataDir, "val2017_IoUs.dat")))
    all_IoAs = np.array(load_data(os.path.join(dataDir, "val2017_IoAs.dat")))
    all_areas_diff = np.array(load_data(os.path.join(dataDir, "val2017_areas_diff.dat")))
    all_areas = np.array(load_data(os.path.join(dataDir, "val2017_object_areas.dat")))

    # Filter out IoUs < 0.3
    IoU_selected = np.logical_and(all_IoUs>0.3, all_IoUs<=1.0)
    all_IoUs = all_IoUs[IoU_selected]
    all_IoAs = all_IoAs[IoU_selected]
    all_areas_diff = all_areas_diff[IoU_selected]
    all_areas = all_areas[IoU_selected]

    # All detected objects
    print("All objects")
    small, large, norm_small, norm_large = analyze_IoU_IoA_areas(all_IoUs,
                                                all_IoAs, all_areas_diff)
    # Definition of object sizes accoridng to https://cocodataset.org/#detection-eval
    # Small detected objects
    print("All small objects")
    S_index = all_areas<math.pow(32,2)
    S_IoUs = all_IoUs[S_index]
    S_IoAs = all_IoAs[S_index]
    S_areas_diff = all_areas_diff[S_index]
    S_obj_small, S_obj_large, S_obj_norm_small, S_obj_norm_large = \
        analyze_IoU_IoA_areas(S_IoUs, S_IoAs, S_areas_diff)
    # Medium detected objects
    print("All medium objects")
    M_index = (all_areas>math.pow(32,2)) & (all_areas<math.pow(96,2))
    M_IoUs = all_IoUs[M_index]
    M_IoAs = all_IoAs[M_index]
    M_areas_diff = all_areas_diff[M_index]
    M_obj_small, M_obj_large, M_obj_norm_small, M_obj_norm_large = \
        analyze_IoU_IoA_areas(M_IoUs, M_IoAs, M_areas_diff)
    # Large detected objects
    print("All large objects")
    L_index = all_areas>math.pow(96,2)
    L_IoUs = all_IoUs[L_index]
    L_IoAs = all_IoAs[L_index]
    L_areas_diff = all_areas_diff[L_index]
    L_obj_small, L_obj_large, L_obj_norm_small, L_obj_norm_large = \
        analyze_IoU_IoA_areas(L_IoUs, L_IoAs, L_areas_diff)

    # Does the obj detector predict small or large with same probability (0.5)?
    # z_test(small, large, "Z-test for all predicitons")
    # z_test(S_obj_small, S_obj_large, "Z-test for all small objects")
    # z_test(M_obj_small, M_obj_large, "Z-test for all mediumn objects")
    # z_test(L_obj_small, L_obj_large, "Z-test for all large objects")

    #%% Group by IoU

    x = np.array([round(i*0.1,1) for i in range(9,2, -1)])

    # Save boxes count
    all_counts, S_counts, M_counts, L_counts = {}, {}, {}, {}
    all_counts['small'], all_counts['large'] = [], []
    all_counts['small_norm'], all_counts['large_norm'] = [], []
    S_counts['small'], S_counts['large'] = [], []
    S_counts['small_norm'], S_counts['large_norm'] = [], []
    M_counts['small'], M_counts['large'] = [], []
    M_counts['small_norm'], M_counts['large_norm'] = [], []
    L_counts['small'], L_counts['large'] = [], []
    L_counts['small_norm'], L_counts['large_norm'] = [], []

    # IoU > 0.9
    all_counts, S_counts, M_counts, L_counts = count_all_boxes(all_IoUs, all_IoAs,
                all_areas_diff, S_IoUs, S_IoAs, S_areas_diff, M_IoUs, M_IoAs,
                M_areas_diff, L_IoUs, L_IoAs, L_areas_diff, all_counts,
                S_counts, M_counts, L_counts, min_IoU = 0.9)
    # # 0.8 < IoU <=0.9
    all_counts, S_counts, M_counts, L_counts = count_all_boxes(all_IoUs, all_IoAs,
                all_areas_diff, S_IoUs, S_IoAs, S_areas_diff, M_IoUs, M_IoAs,
                M_areas_diff, L_IoUs, L_IoAs, L_areas_diff, all_counts,
                S_counts, M_counts, L_counts, min_IoU = 0.8, max_IoU = 0.9)
    # 0.7 < IoU <=0.8# All
    all_counts, S_counts, M_counts, L_counts = count_all_boxes(all_IoUs, all_IoAs,
                all_areas_diff, S_IoUs, S_IoAs, S_areas_diff, M_IoUs, M_IoAs,
                M_areas_diff, L_IoUs, L_IoAs, L_areas_diff, all_counts,
                S_counts, M_counts, L_counts, min_IoU = 0.7, max_IoU = 0.8)
    # 0.6 < IoU <=0.7
    all_counts, S_counts, M_counts, L_counts = count_all_boxes(all_IoUs, all_IoAs,
                all_areas_diff, S_IoUs, S_IoAs, S_areas_diff, M_IoUs, M_IoAs,
                M_areas_diff, L_IoUs, L_IoAs, L_areas_diff, all_counts,
                S_counts, M_counts, L_counts, min_IoU = 0.6, max_IoU = 0.7)
    # 0.5 < IoU <=0.6
    all_counts, S_counts, M_counts, L_counts = count_all_boxes(all_IoUs, all_IoAs,
                all_areas_diff, S_IoUs, S_IoAs, S_areas_diff, M_IoUs, M_IoAs,
                M_areas_diff, L_IoUs, L_IoAs, L_areas_diff, all_counts,
                S_counts, M_counts, L_counts, min_IoU = 0.5, max_IoU = 0.6)
    # 0.4 < IoU <=0.5
    all_counts, S_counts, M_counts, L_counts = count_all_boxes(all_IoUs, all_IoAs,
                all_areas_diff, S_IoUs, S_IoAs, S_areas_diff, M_IoUs, M_IoAs,
                M_areas_diff, L_IoUs, L_IoAs, L_areas_diff, all_counts,
                S_counts, M_counts, L_counts, min_IoU = 0.4, max_IoU = 0.5)
    # 0.3 < IoU <=0.4
    all_counts, S_counts, M_counts, L_counts = count_all_boxes(all_IoUs, all_IoAs,
                all_areas_diff, S_IoUs, S_IoAs, S_areas_diff, M_IoUs, M_IoAs,
                M_areas_diff, L_IoUs, L_IoAs, L_areas_diff, all_counts,
                S_counts, M_counts, L_counts, min_IoU = 0.3, max_IoU = 0.4)

    #%% Plot bb count
    figure_path = os.path.join(dataDir, "bbox_size_count.pdf")
    draw_histogram(x, all_counts['small'], all_counts['large'], figure_path, 6000)

    #%% Plot bb count (normalized)
    figure_path = os.path.join(dataDir, "bbox_size_count_norm.pdf")
    draw_histogram(x, all_counts['small_norm'], all_counts['large_norm'],
                   figure_path, 0.9, True)

    #%% Plot bb count
    # SMALL OBJECTS
    figure_path = os.path.join(dataDir, "SMALL_OBJ_bbox_size_count.pdf")
    draw_histogram(x, S_counts['small'], S_counts['large'], figure_path, 3000)

    #%% Plot bb count (normalized)
    figure_path = os.path.join(dataDir, "SMALL_OBJ_bbox_size_count_norm.pdf")
    draw_histogram(x, S_counts['small_norm'], S_counts['large_norm'],
                   figure_path, 0.9, True)

    #%% Plot bb count
    # MEDIUM OBJECTS
    figure_path = os.path.join(dataDir, "MEDIUM_OBJ_bbox_size_count.pdf")
    draw_histogram(x, M_counts['small'], M_counts['large'], figure_path, 3000)

    #%% Plot bb count (normalized)
    figure_path = os.path.join(dataDir, "MEDIUM_OBJ_bbox_size_count_norm.pdf")
    draw_histogram(x, M_counts['small_norm'], M_counts['large_norm'],
                   figure_path, 0.9, True)

    #%% Plot bb count
    # LARGE OBJECTS
    figure_path = os.path.join(dataDir, "LARGE_OBJ_bbox_size_count.pdf")
    draw_histogram(x, L_counts['small'], L_counts['large'], figure_path, 3000)

    #%% Plot bb count (normalized)
    figure_path = os.path.join(dataDir, "LARGE_OBJ_bbox_size_count_norm.pdf")
    draw_histogram(x, L_counts['small_norm'], L_counts['large_norm'],
                   figure_path, 0.9, True)

    save_data(all_counts, os.path.join(dataDir, "bboxes_count.dat"))
    save_data(S_counts, os.path.join(dataDir, "bboxes_count_small_objects.dat"))
    save_data(M_counts, os.path.join(dataDir, "bboxes_count_medium_objects.dat"))
    save_data(L_counts, os.path.join(dataDir, "bboxes_count_large_objects.dat"))

if __name__ == "__main__":

    datadir = "./IoUs_IoAs_Areas/faster_rcnn_R_50_FPN_3x_bboxes_analysis/"
    main(dataDir = datadir)
