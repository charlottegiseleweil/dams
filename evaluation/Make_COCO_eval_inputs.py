# imports
import argparse
import pandas as pd 
import numpy as np 
import os

# input directories
parser = argparse.ArgumentParser(description='Custom COCO metrics')
parser.add_argument('--images', type=str, default='../../../outputs/yolov3_08-01_detection_results_validation/images', help='directory of images output from detect.py')
parser.add_argument('--predicted_bboxes', type=str, default='../../../outputs/yolov3_08-01_detection_results_validation/pred', help='directory of .txt files of predicted bounding boxes')
parser.add_argument('--ground_truth_bboxes', type=str, default='../../../data/yolov3-inputs_imagery-7-25_cropped_419/validation_set/labels')
args = parser.parse_args()
print(args)

image_ids = os.listdir(os.path.join(args.images))
image_ids.sort()

gt_bbox_fn = os.listdir(os.path.join(args.ground_truth_bboxes))
gt_bbox_fn.sort()

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
        coords = [np.array([x_min, y_min, x_max, y_max])]
        conf = [float('%.4f'%(conf))]
        return coords, conf
    elif format_bbox == 'xywh_norm':
        if 'not_a_dam' in label_fp:
        	coords = [np.array([0, 0, 0, 0])]
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
	            coords = [np.array([x_min, y_min, x_max, y_max])]
	            print(coords)
    return coords

# build ground_truth_bboxes_list and classes_list
classes_list = []
ground_truth_bboxes_list = []
for fn in gt_bbox_fn:
	ground_truth_bboxes_list.append(parse_txt(os.path.join(args.ground_truth_bboxes, fn), 'xywh_norm'))
	classes_list.append(0)
