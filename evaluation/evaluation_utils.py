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
 
    df = make_results_df_per_obs(df_raw,iou_thres,conf_thres)
    
    y_true = np.where(df['gt_bbox'].notnull(), 1, 0)
    y_pred = np.where(df[Category]=='FP', 1,(np.where(df[Category]=='TP',1,0)))
    
    return metrics.recall_score(y_true, y_pred)

def get_precision_score (df_raw, iou_thres=0.05, conf_thres=0.5):
    
    Category = 'Cat@IoU'+str(iou_thres)+'@conf'+str(conf_thres)
    
    try:
        df_raw[Category]
    except KeyError:
        print('df column ' + Category + ' do not exist. Making it...')
        df_raw = add_result_col(df_raw, iou_thres, conf_thres)
        
    df = make_results_df_per_obs(df_raw,iou_thres,conf_thres)
    
    y_true = np.where(df['gt_bbox'].notnull(), 1, 0)
    y_pred = np.where(df[Category]=='FP', 1,(np.where(df[Category]=='TP',1,0)))
    
    return metrics.precision_score(y_true, y_pred)   # NOT average_precision_score

def print_categories_pct(df, iou_thres, conf_thres, pct=True):
    print('with IoU threshold = '+str(iou_thres))
    print('with min. Confidence = '+str(conf_thres))
    df = add_result_col (df, iou_thres, conf_thres)
    Category = 'Cat@IoU'+str(iou_thres)+'@conf'+str(conf_thres)
    count_categories = df[cols+[Category]].groupby(Category).count()['img_id']
    
    # Print absolute numbers of each Category
    if pct!=True:
        print(count_categories,'\n')
    
    # Print % of each category
    elif pct:
        try:
            num_TP = count_categories['TP']/len(df)*100
        except:
            num_TP = 0
        try:
            num_FP = count_categories['FP']/len(df)*100
        except:
            num_FP = 0
        try:
            num_FPFN = count_categories['FP+FN']/len(df)*100
        except:
            num_FPFN = 0

        try:
            num_TN = count_categories['TN']/len(df)*100
        except:
            num_TN = 0
        try:
            num_FN = count_categories['FN']/len(df)*100
        except:
            num_FN = 0

        somme = num_FN+num_FPFN+num_TP+num_FP+num_TN
        if round(somme,0) != 100:
            print('Oops, total is '+str(somme))

        print('TP: {:.0f}%,  FP: {:.1f}%, FP+FN: {:.0f}%, FN: {:.0f}%, TN: {:.0f}%\n'.format(num_TP, num_FP, num_FPFN, num_FN, num_TN))


    
def add_result_col(df, iou_thres, conf_thres):
    """
    Add a column Cat@iou{iou_thres}@conf{conf_thres} to df.
    column contains the category of the image, at this IoU and conf (TP, TN, FP, FN, FN+FP)
    """
    # TP if iou > & conf > (implies GT nontnull and Pred notnull)
    tp_condition = (df.iou > iou_thres) & (df.confidence > conf_thres)
    df['tp'] = np.where(tp_condition, 1, 0)
    
    # TN if gt null & (pred is null | conf <)
    tn_condition = df.gt_bbox.isnull() & ( df.pred_bbox.isnull() | (df.confidence < conf_thres) )
    df['tn'] = np.where(tn_condition, 1, 0)
    
    # FP if gt null & pred (conf>)
    fp_condition = df.gt_bbox.isnull() &  df.pred_bbox.notnull() & (df.confidence > conf_thres) 
    df['fp'] = np.where(fp_condition, 1, 0)
    
    # FN if gt not null & (pred is null | conf <)
    fn_condition =    df.gt_bbox.notnull() & ( df.pred_bbox.isnull() | (df.confidence < conf_thres) ) 
    df['fn'] = np.where(fn_condition, 1, 0)
    
    # FP + FN if if iou < & conf > (implies GT nontnull and Pred notnull)
    fpfn_condition = (df.iou < iou_thres) & (df.confidence > conf_thres)
    df['fpfn'] = np.where(fpfn_condition, 1, 0)
    
    df['Cat@IoU'+str(iou_thres)+'@conf'+str(conf_thres)] = df.apply(
        lambda row: 'Error' if (row.tp + row.tn + row.fp + row.fn + row.fpfn) != 1 \
            else 'TP' if row.tp == 1 \
            else 'TN' if row.tn == 1 \
            else 'FN' if row.fn == 1 \
            else 'FP' if row.fp == 1 \
            else 'FP+FN' if row.fpfn == 1 \
            else 'Error',
            axis=1)
    df = df.drop(['tp', 'fp', 'fn','tn','fpfn'], axis=1)
    
    return df