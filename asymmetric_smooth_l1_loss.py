#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 15:51:20 2021

@author: ombretta
"""


import torch
import math

def asymmetric_smooth_l1_loss(
    input: torch.Tensor, target: torch.Tensor, beta: float, reduction: str = "none",
    asymmetry_factor = 2
) -> torch.Tensor:
    """
    Asymmetric smooth L1 loss proposed to penalize small bounding boxes:
    """
    # Split tensor and then concatenate again to divide x, y, w, h 
    n = input - target
    x_y, w_h = torch.split(n, 2, -1)
    factor = math.sqrt(asymmetry_factor)
    
    if beta < 1e-5:
        # X, Y
        loss_x_y = torch.abs(x_y)
        # w, h
        loss_w_h = torch.where(w_h >= 0, w_h / factor, (-w_h) * factor)
    else:
        # x, y
        cond = x_y < beta
        loss_x_y = torch.where(cond, 0.5 * x_y ** 2 / beta, x_y - 0.5 * beta)
        
        # w, h
        cond2 = (-beta < w_h) & (w_h < 0) # P < G
        cond3 = w_h >= beta # P > G
        cond4 = w_h <= -beta # P < G
        loss_w_h = 0.5 / factor * w_h ** 2 / beta # (0 <= n) & (n < beta) P > G
        loss_w_h = torch.where(cond2, 0.5 * factor * w_h ** 2 / beta, loss_w_h)
        loss_w_h = torch.where(cond3, w_h / factor - 0.5 / factor * beta, loss_w_h)
        loss_w_h = torch.where(cond4, - w_h * factor - 0.5 * factor * beta , loss_w_h)
    
    # Concatenate the loss tensor back
    loss = torch.cat((loss_x_y, loss_w_h), 1)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
