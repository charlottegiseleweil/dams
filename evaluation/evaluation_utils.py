######
#
# Evaluation_Utils.py
#
# Functions needed for Evaluation, done in Evaluation notebook.
# Work on a results dataframe (with cols)
#
# Charlotte Weil, August 2019



# Imports
import os
import pandas as pd
import numpy as np
import random

from sklearn import metrics
import logging


# State variables
cols = ['img_id','gt_bbox','gt_format','pred_bbox','pred_format','confidence','gt_size','iou']


# Functions for Evaluation

def make_results_df_per_obs (results_df, iou_thres=0.05, conf_thres=0.5):
    """
    Transforms results_df (with a row per image), to results_df_obs with a row per observation (duplicate images where FN+FP)
    
    Args:
        results_df is a results dataframe with a row per image (Cateogry column can have value FP+FN)
        iou threshold
        
    Outputs
        results_df_obs is the results-dataframe with a row per observation ((duplicate images where FN+FP))
    """
    
    Category = 'Cat@IoU'+str(iou_thres)+'@conf'+str(conf_thres)
    
    results_df = results_df[cols+[Category]]
    
    
    result_df_slice_OnlyFPFN = results_df[results_df[Category]=='FP+FN']
    
    df_OnlyFP = result_df_slice_OnlyFPFN.copy()
    df_OnlyFN = result_df_slice_OnlyFPFN.copy()
    
    df_OnlyFP[Category] = 'FP'
    df_OnlyFP['gt_bbox'] = None
    
    df_OnlyFN[Category] = 'FN'
    df_OnlyFN['pred_bbox'] = None
    
    result_df_slice_NotFPFN = results_df[results_df[Category]!='FP+FN']
    result_df_obs = result_df_slice_NotFPFN.copy().append(df_OnlyFN).append(df_OnlyFP)
    
    return result_df_obs


def get_recall_score (df_raw, iou_thres=0.05, conf_thres=0.5):
    
    Category = 'Cat@IoU'+str(iou_thres)+'@conf'+str(conf_thres)
    
    try:
        df_raw[Category]
    except KeyError:
        print('df column ' + Category + ' do not exist. Making it...')
        df_raw = add_result_col(df_raw, iou_thres, conf_thres)
    
    if not 'FP+FN' in df_raw[Category].unique():
        raise Exception("The input dataframe must contain a row per image, not a row per observation (thus the raw results df with 'FP+FN'")
    
    df = make_results_df_per_obs(df_raw,iou_thres,conf_thres)
    
    y_true = np.where(df['gt_bbox'].notnull(), 1, 0)
    y_pred = np.where((df[Category]==('TP'or'FP')), 1, 0)
    
    return metrics.recall_score(y_true, y_pred)

def get_precision_score (df_raw, iou_thres=0.05, conf_thres=0.5):
    
    Category = 'Cat@IoU'+str(iou_thres)+'@conf'+str(conf_thres)
    
    try:
        df_raw[Category]
    except KeyError:
        print('df column ' + Category + ' do not exist. Making it...')
        df_raw = add_result_col(df_raw, iou_thres, conf_thres)
    
    if not 'FP+FN' in df_raw[Category].unique():
        raise Exception("The input dataframe must contain a row per image, not a row per observation (thus the raw results df with 'FP+FN'")
    
    df = make_results_df_per_obs(df_raw,iou_thres,conf_thres)
    
    y_true = np.where(df['gt_bbox'].notnull(), 1, 0)
    y_scores = df['confidence'].fillna(0)
    
    return metrics.average_precision_score(y_true, y_scores)


def add_result_col(df, iou_thres, conf_thres):
    """
    Add a column Cat@iou{iou_thres}@conf{conf_thres} to df.
    column contains the category of the image, at this IoU and conf (TP, TN, FP, FN, FN+FP)
    """
    df['tp'] = np.where((df.iou > iou_thres) & (df.confidence > conf_thres), 1, 0)
    df['fp'] = np.where(((df.iou <= iou_thres) | (df.confidence <= conf_thres)) | ((df.pred_bbox.notnull() & df.gt_bbox.isnull())), 1, 0)
    df['fn'] = np.where(((df.iou <= iou_thres) | (df.confidence <= conf_thres)) | ((df.pred_bbox.isnull() & df.gt_bbox.notnull())), 1, 0)
    
    df['Cat@IoU'+str(iou_thres)+'@conf'+str(conf_thres)] = df.apply(
        lambda row: 'TP' if row.tp == 1 else 'FP+FN' if (row.fp == 1) & (row.fn == 1) else 'FP' if (row.fp == 1) else 'FN' if (row.fn == 1) else 'TN',
        axis=1)
    df = df.drop(['tp', 'fp', 'fn'], axis=1)
    
    return df
