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

def parse_txt (label_fp, format_bbox):
    """Obtain x_min, y_min, x_max, and y_max of bounding box from txt file
    Args:
        label_fp (str): filepath to bounding box .txt file in detect.py output format
    Returns:
        coords (list of list)
    """
    if format_bbox == 'x1y1x2y2':
        with open(label_fp, 'r') as label:
            line = str(label.readline())
            vals = line.split(' ')
            x_min = int(vals[0])
            y_min = int(vals[1])
            x_max = int(vals[2])
            y_max = int(vals[3])
            conf = float(vals[5])  
        coords = np.array([[x_min, y_min, x_max, y_max]])
        conf = np.array([float('%.4f'%(conf))])
        return coords, conf
    elif format_bbox == 'xywh_norm':
        if 'not_a_dam' in label_fp:
            coords = np.array([[0, 0, 0, 0]])
            return coords
        else:
            with open(label_fp, 'r') as label_txt:
                line = label_txt.readline()
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
                return coords

# add class label (only one class, so zero) to each list
arr0 = np.array([0])

# build ground_truth_bboxes_list and classes_list
gt_bbox_fn = os.listdir(os.path.join(args.ground_truth_bboxes))
gt_bbox_fn.sort()
gt_img_ids = []
gt_classes_list = []
gt_bboxes_list = []
for fn in gt_bbox_fn:
    gt_img_ids.append(fn)
    gt_bboxes_list.append(parse_txt(os.path.join(args.ground_truth_bboxes, fn), 'xywh_norm'))
    gt_classes_list.append(arr0)

# build predicited_bboxes_list, conf_list, and classes_list
pred_bbox_fn = os.listdir(os.path.join(args.predicted_bboxes))
pred_bbox_fn.sort()
pred_img_ids = []
pred_classes_list = []
pred_conf_list = []
pred_bboxes_list = []
for fn in pred_bbox_fn:
    pred_img_ids.append(fn)
    coords, conf = parse_txt(os.path.join(args.predicted_bboxes, fn), 'x1y1x2y2')
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
) 

# calculate
groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
detections = groundtruth.LoadAnnotations(detections_list)
evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections, agnostic_mode=False)
metrics = evaluator.ComputeMetrics()
print('complete')
