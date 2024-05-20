import os
import numpy as np
from matplotlib import pyplot as plt
import math
import json
import re
from Analyze_IoUs import load_data, save_data
from plot_APs import add_curve


def main():
    plots_path = "plots/"
    datadir = "./IoUs_IoAs_Areas/"

    plt.style.use('ggplot')
    model_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.rc('axes', labelsize=30)
    plt.rc('axes', titlesize=30)
    plt.rc('legend', fontsize=30)
    plt.rc('figure', titlesize=25)

    fig1, ax1 = plt.subplots(figsize=(18, 12))
    models = ["faster_rcnn_R_50_FPN_3x", "retinanet_R_50_FPN_1x", "cascade_mask_rcnn_R_50_FPN_3x"]
    model_names = ["Faster R-CNN", "RetinaNet", "Cascade R-CNN"]
    model_markers = ["o", "*", "X"]
    marker_size = [10, 15, 12]

    x_axis = 'Amount of large detections (%)'
    y_axis = "Average Precision (AP)"
    title = ''
    x_min, x_max = 46, 100
    y_min, y_max = 10, 60

    legend = [r'$\alpha=1$' + " (smooth L1 loss)", r'$\alpha=1.5$', r'$\alpha=2$', r'$\alpha=3$',
              r'$\alpha=4$', r'$\alpha=10$', r'$\alpha=100$']

    for n, model_name in enumerate(models):
        ax1 = add_curve(ax1, [0], [0], model_names[n], y_min, y_max, x_min, x_max,
            title, x_axis, y_axis, fmt=model_markers[n], color='black', marker_size=marker_size[n])

    for n, model_name in enumerate(models):

        all_larges = load_data(datadir + "percent_large_boxes_" + model_name + ".dat")
        all_APs = load_data(datadir + "AP_" + model_name + ".dat")
        all_AP50 = load_data(datadir + "AP50_" + model_name + ".dat")

        legend_AP = "" if n < 2 else "AP"
        legend_AP50 = "" if n < 2 else "AP50"

        ax1 = add_curve(ax1, all_larges, all_APs, legend_AP, y_min, y_max, x_min, x_max,
                        title, x_axis, y_axis, fmt="-.", color='#898A8A')
        ax1 = add_curve(ax1, all_larges, all_AP50, legend_AP50, y_min, y_max, x_min, x_max,
                        title, x_axis, y_axis, fmt=':', color='#898A8A')

        for idx, alpha in enumerate(legend):
            if n > 0: alpha = ''
            ax1 = add_curve(ax1, all_larges[idx], all_APs[idx], alpha, y_min, y_max, x_min, x_max, title, x_axis,
                            y_axis, fmt=model_markers[n], color=model_colors[idx], marker_size=marker_size[n])
            ax1 = add_curve(ax1, all_larges[idx], all_AP50[idx], "", y_min, y_max, x_min, x_max, title, x_axis,
                            y_axis, fmt=model_markers[n], color=model_colors[idx], marker_size=marker_size[n])




    ax1.tick_params(axis='both', which='major', labelsize=25)
    ax1.legend(bbox_to_anchor=(-0.0, -0.28), loc='lower left', ncol=4, fontsize=22)
    fig1.savefig(plots_path + "largebboxes_AP.pdf", dpi=100, bbox_inches='tight')

if __name__ == "__main__":
    main()