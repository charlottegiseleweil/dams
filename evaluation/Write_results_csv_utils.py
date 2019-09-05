"""


 Write_results_csv.py

 (Just the function to be called in notebooks - not the args - can't call this script from command line.)
 
 Creates results dataframe from directories and format for ground truth boxes and predicted boxes. 
    done in Evaluation notebook.
 
 Charlotte Weil, August 2019

"""


# import required modules

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from shapely.geometry import box
import random
from sklearn import metrics

import humanfriendly
import time


# Utils
def parse_txt (label_fp, format_bbox, gt_or_pred):
    """Obtain x_min, y_min, x_max, and y_max of bounding box from txt file
    Args:
        label_fp (str): filepath to bounding box .txt file in detect.py output format
        format_bbox: 'x1y1x2y2' or 'xywh_pix' or 'x1y1x2y2_pix'
        gt_or_pred: 'predicted' or 'ground_truth'
    Returns:
        coords (numpy array)
        conf (float, returned only if dataset == 'predicted')
    """
    if format_bbox == 'y1x1y2x2_pix':
        with open(label_fp, 'r') as label:
            line = str(label.readline())
            if len(line) == 0:
                return None
            else:
                vals = line.split(' ')
                x_min = int(float(vals[1]) * 419)
                y_min = int(float(vals[2]) * 419)
                x_max = int(float(vals[3]) * 419)
                y_max = int(float(vals[0]) * 419)
                coords = np.array([y_min, x_min, y_max, x_max])
                if gt_or_pred == 'ground_truth':
                    return coords
                elif gt_or_pred == 'predicted':
                    conf = float(vals[5])
                    conf = '%.4f'%(conf)
                    return coords, conf
    elif format_bbox == 'xywh_pix':
        with open(label_fp, 'r') as label:
            line = str(label.readline())
            if len(line) == 0:
                return None
            else:
                vals = line.split(' ')
                norm_x = float(vals[1])
                norm_y = float(vals[2])
                norm_w = float(vals[3])
                norm_h = float(vals[4])
                x_min = int((norm_x * 419) - ((norm_w * 419) / 2))
                y_min = int((norm_y * 419) - ((norm_h * 419) / 2))
                x_max = int((norm_x * 419) + ((norm_w * 419) / 2))
                y_max = int((norm_y * 419) + ((norm_h * 419) / 2))
                coords = np.array([x_min, y_min, x_max, y_max])
                if gt_or_pred == 'ground_truth':
                    return coords
                elif gt_or_pred == 'predicted':
                    conf = float(vals[5])
                    conf = '%.4f'%(conf)
                    return coords, conf
                

def calc_IoU (bb1, bb2, gt_format, pred_format):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Adapted from: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Args:
        bb1: [x1,y1,x2,y2]
        bb2: [x1,y1,x2,y2]    
    The (x1, y1) position is at the top left corner (or the bottom right - either way works).
    The (x2, y2) position is at the bottom right corner (or the top left).
    Returns:
        intersection_over_union, a float in [0, 1]
    """
    
    # convert to x1y1x2y2 format if needed
    if gt_format == 'y1x1y2x2_pix':
        y_max, x_min, y_min, x_max = bb1
        bb1 = [x_min, y_min, x_max, y_max]
    if pred_format == 'y1x1y2x2_pix':
        y_max, x_min, y_min, x_max = bb2
        bb2 = [x_min, y_min, x_max, y_max]

    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    
    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area.
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    
    return iou



def make_results_table (gt_bbox_dir, pred_bbox_dir, outputFile,
                        gt_format='xywh_pix', pred_format='y1x1y2x2_pix'):
    '''
    Creates dataframe from directories and format for ground truth boxes and predicted boxes. 
    Args:
        gt_bbox_dir: directory of ground truth bounding boxes        
        pred_bbox_dir: directory of ground truth bounding boxes
        
        outputFile : filepath to .csv results table to write (results_{set}_{model}.csv)
        
        
        gt_format (Optional): format of ground truth bounding boxes 'y1x1y2x2_pix' or 'xywh_pix'. Default is 'xywh_pix'
        pred_format (Optional): format of ground truth bounding boxes 'y1x1y2x2_pix' or 'xywh_pix' . Default is 'y1x1y2x2_pix'
    Returns:
        dataframe with cols:
        ['img_id','gt_bbox','gt_format','pred_bbox','pred_format','confidence','gt_size','iou']

    '''
    
    print('Loading GT & predicted bboxes...')
    startTime = time.time()
    
    # collect list of filenames
    pred_bbox_fn = [fn[:-4] for fn in os.listdir(pred_bbox_dir)]
    gt_bbox_fn = [fn[:-4] for fn in os.listdir(gt_bbox_dir)]
    
    elapsed = time.time() - startTime
    print("Finished loaded {} GT bboxes and {} predicted bboxes in {}".format(len(gt_bbox_fn),len(pred_bbox_fn),
          humanfriendly.format_timespan(elapsed)))
    
    # create dictionaries of ground_truth bboxes,  predicted bboxes, and confidence
    ground_truth_dict = {}
    for fn in gt_bbox_fn:
        bbox = parse_txt(os.path.join(gt_bbox_dir, fn+'.txt'), gt_format, 'ground_truth')
        ground_truth_dict[fn] = bbox
    predicted_dict = {}
    conf_dict = {}
    for fn in pred_bbox_fn:
        bbox, conf = parse_txt(os.path.join(pred_bbox_dir, fn+'.txt'), pred_format, 'predicted')
        predicted_dict[fn] = bbox
        conf_dict[fn] = conf
    
    # build dataframe of image, ground_truth bboxes, predicted bboxes, and confidence
    print('Creating results table with ...')
    data = {'img_id' : gt_bbox_fn[:-4]}
    detect_df = pd.DataFrame(data, columns=['img_id'])
    
    print('... GT bboxes')
    detect_df['gt_bbox'] = detect_df['img_id'].map(pd.Series(ground_truth_dict))
    detect_df['gt_format'] = gt_format
    
    print('... predicted bboxes')
    detect_df['pred_bbox'] = detect_df['img_id'].map(pd.Series(predicted_dict))
    detect_df['pred_format'] = pred_format
    
    print('... confidence')
    detect_df['confidence'] = detect_df['img_id'].map(pd.Series(conf_dict))
    
    # add ground_truth bbox size to dataframe
    print('... Bbox size')
    detect_df['gt_size'] = detect_df.apply(
        lambda row: None if row.gt_bbox is None else int((row.gt_bbox[2] - row.gt_bbox[0]) * (row.gt_bbox[3] - row.gt_bbox[1])), 
        axis=1
    )

    
    # add IoU - cameratraps to dataframe
    
    print('... IoU')
    iou_dict = {}
    for fn in predicted_dict:
        if 'not_a_dam' not in fn:
            iou = calc_IoU(ground_truth_dict[fn], predicted_dict[fn], gt_format, pred_format)
            iou_dict[fn] = iou
        else:
            iou_dict[fn] = None
    detect_df['iou'] = detect_df['img_id'].map(pd.Series(iou_dict))
    
    print('... Done \n')

    
    
    # set index to img_id
    detect_df = detect_df.set_index('img_id')
    
    print('Writing CSV ...')
    detect_df.to_csv(outputFile)
    print('... Done')
    
    return detect_df