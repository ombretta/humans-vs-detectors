#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:40:22 2021

@author: ombretta
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from Analyze_IoUs import load_data, save_data
import re

def find_AP_from_text(path, dataset):
    with open(path, "r") as f:
        full_text = f.read()
    
    text_to_find = r" APl   \|\n.*--:\|\n\| "
    r1 = re.search(text_to_find,full_text) 
    index = r1.end()
    # print("AP", index)
    # AP = full_text[index:index+6]
    AP, AP50, AP75, APs, APm, APl = full_text[index:index+55].split(" | ")[:6]
    APl = APl.split(" ")[0]
    # print(AP)
    return float(AP), float(AP50), float(AP75), float(APs), float(APm), float(APl)


def add_curve(ax, x, y, label, y_min, y_max, x_min, x_max, title,
              x_label, y_label, fmt='-o', color='', marker_size=10):
    ax.errorbar(x, y, fmt=fmt, label=label, color=color, markersize=marker_size, linewidth=3)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)   
    ax.legend()
    return ax 

def add_curve_with_std(fig, ax, x, y, x_err, y_err, label, y_min, y_max, x_min, x_max, 
              title, x_label, y_label, color='', fmt='-o', marker_size=10):
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=fmt, label=label, color=color, markersize=marker_size)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)   
    ax.legend()
    return ax 

def main():
    
    root_path = "/Users/ombretta/Documents/Code/evaluating_IoU/tested_models/"+\
        "coco/ROI_HEADS_SCORE_THRESH_TEST_0.5/"
    plots_path = root_path+"plots/"
    
    if not os.path.exists(plots_path): os.mkdir(plots_path)
    
    # model_name = "faster_rcnn_R_50_FPN_3x"
    # model_name = "retinanet_R_50_FPN_1x"
    model_name = "cascade_mask_rcnn_R_50_FPN_3x"
    # model_name = "faster_rcnn_R_50_C4"
    
    model_root = model_name + "_detectron2_pretrained"
    
    factors = ["", 1.5, 2, 3, 4, 10, 100]
    beta = [""]#, 1.0, 0.1]
    
    models_res = {}
    
    model_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    
    plt.rc('axes', labelsize=20) 
    plt.rc('axes', titlesize=25)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=25) 
    
    # PLOT AP OVER ITERATIONS
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    fig1_50, ax1_50 = plt.subplots(figsize=(12, 8))
    fig1_75, ax1_75 = plt.subplots(figsize=(12, 8))
    fig1_s, ax1_s = plt.subplots(figsize=(12, 8))
    fig1_m, ax1_m = plt.subplots(figsize=(12, 8))
    fig1_l, ax1_l = plt.subplots(figsize=(12, 8))
    
    
    xlabel = 'Fine-tuning iterations'
    x_min, x_max = 0, 120000
    
    idx = 0
    for f in factors:
        model = model_root + "_asymmetric_smooth_l1_factor"+str(f) if f != "" else model_root
        for b in beta:
            model_beta = model + "_beta"+str(b) if b != "" else model
            
            x_axis = []
            all_AP, all_AP50, all_AP75, all_APs, all_APm, all_APl = [], [], [], [], [], []
            
            models = [m for m in os.listdir(root_path) if ((model_beta + "_lr" in m \
                      and "model_final" not in m) or "detectron2_pretrained_bboxes_analysis" in m)]
            models = sorted(models)

            print(idx, "MODELS", models)
            for m in models:
                print(m)
                iters_index = m.find("_lr0.00025_model_")+len("_lr0.00025_model_")
                iters = int(m[iters_index:iters_index+7])+1 if int(m[iters_index:iters_index+7]) != 0 else 0
                if iters <= 100000:
                    x_axis.append(iters)
                    AP, AP50, AP75, APs, APm, APl = find_AP_from_text(root_path+m, 'coco')
                    print("AP", AP)
                    all_AP.append(AP)
                    all_AP50.append(AP50)
                    all_AP75.append(AP75)
                    all_APs.append(APs)
                    all_APm.append(APm)
                    all_APl.append(APl)
                    if model_beta not in models_res: models_res[model_beta] = {}
                    if iters not in models_res[model_beta]:
                        models_res[model_beta][iters] = {}
                    models_res[model_beta][iters]["AP"] = AP
                    models_res[model_beta][iters]["AP50"] = AP50
                    models_res[model_beta][iters]["AP75"] = AP75
                    models_res[model_beta][iters]["APs"] = APs
                    models_res[model_beta][iters]["APm"] = APm
                    models_res[model_beta][iters]["APl"] = APl
            
            
            label = model_beta 
            print("legend", label)
            print("idx", idx, "model_colors", model_colors)

            if len(all_AP)>0:
                
                all_AP = [all_AP for _,all_AP in sorted(zip(x_axis,all_AP))]
                all_AP50 = [all_AP50 for _,all_AP50 in sorted(zip(x_axis,all_AP50))]
                all_AP75 = [all_AP75 for _,all_AP75 in sorted(zip(x_axis,all_AP75))]
                all_APs = [all_APs for _,all_APs in sorted(zip(x_axis,all_APs))]
                all_APm = [all_APm for _,all_APm in sorted(zip(x_axis,all_APm))]
                all_APl = [all_APl for _,all_APl in sorted(zip(x_axis,all_APl))]
                x_axis = sorted(x_axis)
                ax1 = add_curve(ax1, x_axis, all_AP, label, 16, 40, 
                    x_min, x_max, model_name, xlabel, "AP", '-o', model_colors[idx])
                ax1_50 = add_curve(ax1_50, x_axis, all_AP50, label, 30, 60, 
                    x_min, x_max, model_name, xlabel, "AP50", '-o', model_colors[idx])
                ax1_75 = add_curve(ax1_75, x_axis, all_AP75, label, 0, 50, 
                    x_min, x_max, model_name, xlabel, "AP75", '-o', model_colors[idx])
                ax1_s = add_curve(ax1_s, x_axis, all_APs, label, 0, 30 ,
                    x_min, x_max, model_name, xlabel, "APs", '-o', model_colors[idx])
                ax1_m = add_curve(ax1_m, x_axis, all_APm, label, 0, 60, 
                    x_min, x_max, model_name, xlabel, "APm", '-o', model_colors[idx])
                ax1_l = add_curve(ax1_l, x_axis, all_APl, label, 0, 60, 
                    x_min, x_max, model_name, xlabel, "APl", '-o', model_colors[idx])
        idx += 1
                
    fig1.savefig(plots_path + "finetuning_curves_AP_" + model_name + ".png")
    fig1_50.savefig(plots_path + "finetuning_curves_AP50_" + model_name + ".png")
    fig1_75.savefig(plots_path + "finetuning_curves_AP75_" + model_name + ".png")
    fig1_s.savefig(plots_path + "finetuning_curves_APs_" + model_name + ".png")
    fig1_m.savefig(plots_path + "finetuning_curves_APm_" + model_name + ".png")
    fig1_l.savefig(plots_path + "finetuning_curves_APl_" + model_name + ".png")
    
    # PLOT % LARGE BOXES OVER ITERATIONS
    
    datadir = "./IoUs_IoAs_Areas/"
    plot_path2 = plots_path + "largebboxes_finetuning_" + model_name + ".png"
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    idx = 0
    for f in factors:
        model = model_root + "_asymmetric_smooth_l1_factor"+str(f) if f != "" else model_root
        for b in beta:
            model_beta =  model + "_beta"+str(b) if b != "" else model
            x_axis = []
            larges = []
            models = [m for m in os.listdir(datadir) if model_beta + "_lr" in m \
                      and os.path.isdir(datadir+m)]
            # models = [m for m in os.listdir(datadir) if model_beta in m \
            #           and os.path.isdir(datadir + m)]
            models = sorted(models)
            
            for m in models:
                print("checking", m)
                # if "k_iters" in m and os.path.exists(datadir+m+"/bboxes_count.dat"):
                if os.path.exists(datadir + m + "/bboxes_count.dat"):
                    print(m)
                    iters = int(m.split("k_iters")[0].split("_")[-1])*1000
                    x_axis.append(int(iters))
                    bboxes_count = load_data(datadir+m+"/bboxes_count.dat")
                    large = bboxes_count['large']
                    small = bboxes_count['small']
                    large_perc = round(sum(large)/(sum(large)+sum(small))*100, 2)
                    print(large_perc)
                    larges.append(large_perc)
                    if model_beta not in models_res: models_res[model_beta] = {}
                    if iters not in models_res[model_beta]:
                        models_res[model_beta][iters] = {}
                    models_res[model_beta][iters]["large_bboxes%"] = large_perc
                    
            if len(larges):
                
                larges = [larges for _,larges in sorted(zip(x_axis,larges))]
                x_axis = sorted(x_axis)
                print(x_axis)
                print(larges)
                
                ax2 = add_curve(ax2, x_axis, larges, model_beta, 0, 100, x_min, x_max, 
                            model_name, 'Fine-tuning iterations', "% large bboxes",
                            '-o', model_colors[idx])
                idx += 1
        
    plt.savefig(plot_path2)
    
    #  PLOT AP OVER % LARGE BOXES OVER ITERATIONS
    
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    fig4, ax4 = plt.subplots(figsize=(18, 12))
    
    all_APs, all_AP50, all_AP75, all_APss, all_APm, all_APl = [], [], [], [], [], []
    all_larges = []
    x_min, x_max = 42, 109
    idx = 0

    print("MODELS RESULTS", models_res)

    for model in models_res:
        AP, AP50, AP75, APs, APm, APl = [], [], [], [], [], []
        larges = []
        print(sorted(list(models_res[model].keys())))
        for iters in sorted(list(models_res[model].keys())):
            res = models_res[model][iters]
            if 'AP' in res and 'large_bboxes%' in res:
                AP.append(res["AP"])
                AP50.append(res["AP50"])
                AP75.append(res["AP75"])
                APs.append(res["APs"])
                APm.append(res["APm"])
                APl.append(res["APl"])
                larges.append(res["large_bboxes%"])
        print("AP", AP)
        AP = AP[-1:] # to plot only one point per model (one chosen iteration)
        AP50 = AP50[-1:]
        AP75 = AP75[-1:]
        APs = APs[-1:]
        APm = APm[-1:]
        APl = APl[-1:]
        larges = larges[-1:]

        all_APs += AP
        all_AP50 += AP50
        all_AP75 += AP75
        all_APss += APs
        all_APm += APm
        all_APl += APl
        all_larges += larges
        
        legend = "Factor "
        legend += model.split("factor")[-1] if 'factor' in model else "1\n(smooth L1 loss)" 
        
        if len(larges):    
            ax3 = add_curve(ax3, larges, AP, legend, 0, 50, x_min, x_max, 
                            model_name, '% large bboxes', "AP", '-o', model_colors[idx])
            
            ax4 = add_curve_with_std(fig4, ax4, np.mean(larges), np.mean(AP), 
                     np.std(larges), np.std(AP), legend, 25, 40, x_min, x_max, 
                     model_name, '% large bboxes', "AP", model_colors[idx])
            ax4 = add_curve_with_std(fig4, ax4, np.mean(larges), np.mean(AP50), 
                     np.std(larges), np.std(AP50), "", 25, 40, x_min, x_max, 
                     model_name, '% large bboxes', "AP", model_colors[idx])
            ax4 = add_curve_with_std(fig4, ax4, np.mean(larges), np.mean(AP75), 
                     np.std(larges), np.std(AP75), "", 25, 40, x_min, x_max, 
                     model_name, '% large bboxes', "AP", model_colors[idx])
            ax4 = add_curve_with_std(fig4, ax4, np.mean(larges), np.mean(APs), 
                     np.std(larges), np.std(APs), "", 25, 40, x_min, x_max, 
                     model_name, '% large bboxes', "AP", model_colors[idx])
            ax4 = add_curve_with_std(fig4, ax4, np.mean(larges), np.mean(APm), 
                     np.std(larges), np.std(APm), "", 25, 40, x_min, x_max, 
                     model_name, '% large bboxes', "AP", model_colors[idx])
            ax4 = add_curve_with_std(fig4, ax4, np.mean(larges), np.mean(APl), 
                     np.std(larges), np.std(APl), "", 25, 40, x_min, x_max, 
                     model_name, '% large bboxes', "AP", model_colors[idx])
            idx += 1
            
    
    all_APs = [all_APs for _,all_APs in sorted(zip(all_larges,all_APs))]
    all_AP50 = [all_AP50 for _,all_AP50 in sorted(zip(all_larges,all_AP50))]
    all_AP75 = [all_AP75 for _,all_AP75 in sorted(zip(all_larges,all_AP75))]
    all_APss = [all_APss for _,all_APss in sorted(zip(all_larges,all_APss))]
    all_APm = [all_APm for _,all_APm in sorted(zip(all_larges,all_APm))]
    all_APl = [all_APl for _,all_APl in sorted(zip(all_larges,all_APl))]
    all_larges = sorted(all_larges)
    
    title = " ".join(model_name.split("_")).upper()
    y_min, y_max = 0, 55
    ax4 = add_curve(ax4, all_larges, all_APs, "AP", y_min, y_max, x_min, x_max, 
        title, '% large bboxes', "AP", fmt=":", color='#181A1A')
    ax4 = add_curve(ax4, all_larges, all_AP50, "AP50", y_min, y_max, x_min, x_max, 
        title, '% large bboxes', "AP", fmt=":", color='#81F6F6')
    ax4 = add_curve(ax4, all_larges, all_AP75, "AP75", y_min, y_max, x_min, x_max, 
        title, '% large bboxes', "AP", fmt=":",  color='#0A9595')
    ax4 = add_curve(ax4, all_larges, all_APss, "APs", y_min, y_max, x_min, x_max, 
        title, '% large bboxes', "AP", fmt=":", color='#E0B6DE')
    ax4 = add_curve(ax4, all_larges, all_APm, "APm", y_min, y_max, x_min, x_max, 
        title, '% large bboxes', "AP", fmt=":", color='#E570E0')
    ax4 = add_curve(ax4, all_larges, all_APl, "APl", y_min, y_max, x_min, x_max, 
        title, '% large bboxes', "AP", fmt=":", color='#8C0887')

    fig3.savefig(plots_path + "largebboxes_AP_" + model_name + ".png")
    fig4.savefig(plots_path + "largebboxes_all_APs_" + model_name + ".png")
    plt.show()

    save_data(all_larges, datadir + "percent_large_boxes_" + model_name + ".dat")
    save_data(all_APs, datadir + "AP_" + model_name + ".dat")
    save_data(all_AP50, datadir + "AP50_" + model_name + ".dat")
    save_data(all_AP75, datadir + "AP75_" + model_name + ".dat")
    save_data(all_APss, datadir + "APs_" + model_name + ".dat")
    save_data(all_APm, datadir + "APm_" + model_name + ".dat")
    save_data(all_APl, datadir + "APl_" + model_name + ".dat")

    
    
            
if __name__ == "__main__":
    main()    
            