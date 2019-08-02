''' 
Prepares input for object_detection/metrics/coco_eval.py 
Runs object_detection/metrics/coco_eval.py to calculate COCO metrics

calc_coco_metrics.py usage:
    Inputs:
        directory of predicted bounding boxes as txts
            txt format may be 'x1y1x2y2' or 'xywh_norm'
            <x1> <y1> <x2> <y2> <class> <confidence> (separated by spaces, all ints except for confidence, which is float)
            <x-norm> <y-norm> <w-norm> <h-norm> <class> <confidence> (separated by spaces, all floats except for class, which is int)
        directory of ground truth bounding boxea as txts
            txt format may be 'x1y1x2y2' or 'xywh_norm'
            <x1> <y1> <x2> <y2> <class> <confidence> (separated by spaces, all ints except for confidence, which is float)
            <x-norm> <y-norm> <w-norm> <h-norm> <class> <confidence> (separated by spaces, all floats except for class, which is int)
    Outputs:
        COCO metrics printed to console (by calling coco_eval.py)

coco_eval.py usage:

    Given a set of images with ids in the list image_ids
    and corresponding lists of numpy arrays encoding groundtruth (boxes and classes)
    and detections (boxes, scores and classes), where elements of each list
    correspond to detections/annotations of a single image,
    then evaluation (in multi-class mode) can be invoked as follows:
      groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(
          image_ids, groundtruth_boxes_list, groundtruth_classes_list,
          max_num_classes, output_path=None)
      detections_list = coco_tools.ExportDetectionsToCOCO(
          image_ids, detection_boxes_list, detection_scores_list,
          detection_classes_list, output_path=None)
      groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
      detections = groundtruth.LoadAnnotations(detections_list)
      evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,
                                             agnostic_mode=False)
      metrics = evaluator.ComputeMetrics()
  
'''

# imports
import argparse
import pandas as pd 
import numpy as np 
import os
from object_detection.metrics import coco_tools

# input directories
parser = argparse.ArgumentParser(description='Custom COCO metrics')
parser.add_argument('--predicted_bboxes', type=str, default='../../../outputs/yolov3_08-01_detection_results_validation/pred', help='directory of .txt files of predicted bounding boxes')
parser.add_argument('--ground_truth_bboxes', type=str, default='../../../data/yolov3-inputs_imagery-7-25_cropped_419/validation_set/labels')
args = parser.parse_args()
print(args)

def parse_txt (label_fp, format_bbox, dataset):
    """Obtain x_min, y_min, x_max, and y_max of bounding box from txt file
    Args:
        label_fp (str): filepath to bounding box .txt file in detect.py output format
        format_bbox: 'x1y1x2y2' or 'xywh_norm'
        dataset: 'predicted' or 'ground_truth'
    Returns:
        coords (numpy array of shape [1, 4])
        conf (numpy array of shape [1], returned only if dataset == 'predicted')
    """
    if format_bbox == 'x1y1x2y2':
        with open(label_fp, 'r') as label:
            line = str(label.readline())
            vals = line.split(' ')
            x_min = int(vals[0])
            y_min = int(vals[1])
            x_max = int(vals[2])
            y_max = int(vals[3])
        coords = np.array([[x_min, y_min, x_max, y_max]])
        if dataset == 'ground_truth':
            return coords
        elif dataset == 'predicted':
            conf = float(vals[5])
            conf = np.array([float('%.4f'%(conf))])
            return coords, conf
    elif format_bbox == 'xywh_norm':
        with open(label_fp, 'r') as label:
            line = label.readline()
            vals = line.split(' ')
            norm_x = float(vals[1])
            norm_y = float(vals[2])
            norm_w = float(vals[3])
            norm_h = float(vals[4])
            x_min = int((norm_x * 419) - ((norm_w * 419) / 2))
            y_min = int((norm_y * 419) - ((norm_h * 419) / 2))
            x_max = int((norm_x * 419) + ((norm_w * 419) / 2))
            y_max = int((norm_y * 419) + ((norm_h * 419) / 2))
            coords = np.array([[x_min, y_min, x_max, y_max]])
        if dataset == 'ground_truth':
            return coords
        elif dataset == 'predicted':
            conf = float(vals[5])
            conf = np.array([float('%.4f'%(conf))])
            return coords, conf

# add class label (only one class, so zero) to each list
arr0 = np.array([0])

# build ground_truth_bboxes_list and classes_list
gt_bbox_fn = os.listdir(os.path.join(args.ground_truth_bboxes))
gt_bbox_fn.sort()
gt_img_ids = []
gt_classes_list = []
gt_bboxes_list = []
for fn in gt_bbox_fn:
    if 'not_a_dam' not in fn:
        gt_img_ids.append(fn)
        gt_bboxes_list.append(parse_txt(os.path.join(args.ground_truth_bboxes, fn), 'xywh_norm', 'ground_truth'))
        gt_classes_list.append(arr0)

# build predicited_bboxes_list, conf_list, and classes_list
pred_bbox_fn = os.listdir(os.path.join(args.predicted_bboxes))
pred_bbox_fn.sort()
pred_img_ids = []
pred_classes_list = []
pred_conf_list = []
pred_bboxes_list = []
for fn in pred_bbox_fn:
    if 'not_a_dam' not in fn:
        pred_img_ids.append(fn)
        coords, conf = parse_txt(os.path.join(args.predicted_bboxes, fn), 'x1y1x2y2', 'predicted')
        pred_bboxes_list.append(coords)
        pred_conf_list.append(conf)
        pred_classes_list.append(arr0)

# add category input
categories = [{'id' : 0, 'name' : 'dam'}]

# prepare ground_truth input for COCOWrapper
groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(
    gt_img_ids, 
    gt_bboxes_list, 
    gt_classes_list,
    categories, 
)    

# prepare detections input for COCOWrapper
detections_list = coco_tools.ExportDetectionsToCOCO(
    pred_img_ids, 
    pred_bboxes_list, 
    pred_conf_list, 
    pred_classes_list,
    categories
) 

# calculate
groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
detections = groundtruth.LoadAnnotations(detections_list)
evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections, agnostic_mode=False)
metrics = evaluator.ComputeMetrics()
print('complete')
