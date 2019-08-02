# imports
import argparse
import os
import pandas as pd
import numpy as np

# inputs
parser = argparse.ArgumentParser(description='Custom COCO metrics')
parser.add_argument('--images', type=str, default='../../../outputs/yolov3_08-01_detection_results_validation/images', help='directory of images output from detect.py')
parser.add_argument('--predicted_bboxes', type=str, default='../../../outputs/yolov3_08-01_detection_results_validation/pred/', help='directory of .txt files of predicted bounding boxes')
parser.add_argument('--predicted_format', type=str, default='x1y1x2y2', help='txt format for predicted bbox files')
parser.add_argument('--ground_truth_bboxes', type=str, default='../../../data/yolov3-inputs_imagery-7-25_cropped_419/validation_set/labels/')
parser.add_argument('--ground_truth_format', type=str, default='xywh_norm', help='txt format for ground_truth bbox files')
parser.add_argument('--iou_thres', type=float, default=0.5, help='IoU threshold for mAP calculation')
parser.add_argument('--output_path', type=str, default='results_validation_fasterRCNN-07-27_IoU2.csv', help='path for output csv')
args = parser.parse_args()
print(args)
# set directory paths
images_dir = os.path.join(args.images)
pred_bbox_dir = os.path.join(args.predicted_bboxes)
gt_bbox_dir = os.path.join(args.ground_truth_bboxes)

# obtain lists of files
image_fn = os.listdir(images_dir)
pred_bbox_fn = os.listdir(pred_bbox_dir)
gt_bbox_fn = os.listdir(gt_bbox_dir)

# constants
iou_thres = args.iou_thres

def parse_txt (label_fp, format_bbox, dataset):
	"""Obtain x_min, y_min, x_max, and y_max of bounding box from txt file
	Args:
		label_fp (str): filepath to bounding box .txt file in detect.py output format
		format_bbox: 'x1y1x2y2' or 'xywh_norm' or 'x1y1x2y2_norm'
		dataset: 'predicted' or 'ground_truth'
	Returns:
		coords (numpy array)
		conf (float, returned only if dataset == 'predicted')
	"""
	if 'not_a_dam' in label_fp:
		if dataset == 'ground_truth':
			return np.array([0, 0, 0, 0])
		else:
			return np.array([1, 1, 1, 1]), float(0)
	else:
		if format_bbox == 'x1y1x2y2':
			with open(label_fp, 'r') as label:
				line = str(label.readline())
				vals = line.split(' ')
				x_min = int(vals[0])
				y_min = int(vals[1])
				x_max = int(vals[2])
				y_max = int(vals[3])
			coords = np.array([y_min, x_min, y_max, x_max])
			if dataset == 'ground_truth':
				return coords
			elif dataset == 'predicted':
				conf = float(vals[5])
				conf = '%.4f'%(conf)
				return coords, conf
		if format_bbox == 'x1y1x2y2_norm':
			with open(label_fp, 'r') as label:
				line = str(label.readline())
				vals = line.split(' ')
				x_min = int(float(vals[0]) * 419)
				y_min = int(float(vals[1]) * 419)
				x_max = int(float(vals[2]) * 419)
				y_max = int(float(vals[3]) * 419)
			coords = np.array([y_min, x_min, y_max, x_max])
			if dataset == 'ground_truth':
				return coords
			elif dataset == 'predicted':
				conf = float(vals[5])
				conf = '%.4f'%(conf)
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
				coords = np.array([y_min, x_min, y_max, x_max])
			if dataset == 'ground_truth':
				return coords
			elif dataset == 'predicted':
				conf = float(vals[5])
				conf = '%.4f'%(conf)
				return coords, conf

def calc_IoU (ground_truth, predicted):
	"""Calculate IoU of single predicted and ground truth box
	Args:
		pred_box (list of ints): location of predicted object as [xmin, ymin, xmax, ymax]
		gt_box (list of ints): location of ground truth object as [xmin, ymin, xmax, ymax]
	Returns:
		float: value of the IoU for the two boxes.
	"""
	# gather gt and pred coords
	x1_t, y1_t, x2_t, y2_t = ground_truth
	x1_p, y1_p, x2_p, y2_p = predicted
	# find extremes
	far_x = np.min([x2_t, x2_p])
	near_x = np.max([x1_t, x1_p])
	far_y = np.min([y2_t, y2_p])
	near_y = np.max([y1_t, y1_p])
	# calc areas
	inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
	gt_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
	pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
	# calc IoU
	iou = inter_area / (gt_box_area + pred_box_area - inter_area)
	iou = float('%.4f'%(iou))
	return iou

# create dictionaries of ground_truth bboxes,  predicted bboxes, and confidence
ground_truth_dict = {}
for fn in gt_bbox_fn:
	bbox = parse_txt(os.path.join(gt_bbox_dir, fn), args.ground_truth_format, 'ground_truth')
	ground_truth_dict[fn.replace('.txt', '.png')] = bbox
predicted_dict = {}
conf_dict = {}
for fn in pred_bbox_fn:
	bbox, conf = parse_txt(os.path.join(pred_bbox_dir, fn), args.predicted_format, 'predicted')
	predicted_dict[fn.replace('.txt', '.png')] = bbox
	conf_dict[fn.replace('.txt', '.png')] = conf

# build dataframe of image, ground_truth bboxes, predicted bboxes, and confidence
data = {'image' : image_fn}
detect_df = pd.DataFrame(data, columns=['image'])
print('created dataframe')
detect_df['ground_truth'] = detect_df['image'].map(pd.Series(ground_truth_dict))
print('added ground truth boxes')
detect_df['predicted'] = detect_df['image'].map(pd.Series(predicted_dict))
print('added predicted boxes')
detect_df['confidence'] = detect_df['image'].map(pd.Series(conf_dict))
print('added confidence')

# add ground_truth bbox size to dataframe
detect_df['size'] = detect_df.apply(
	lambda row: int((row.ground_truth[2] - row.ground_truth[0]) * (row.ground_truth[3] - row.ground_truth[1])), 
	axis=1
)
print('added size')

# add IoU to dataframe
iou_dict = {}
for fn in predicted_dict:
	iou = calc_IoU(ground_truth_dict[fn], predicted_dict[fn])
	iou_dict[fn] = iou
detect_df['iou'] = detect_df['image'].map(pd.Series(iou_dict))
print('added iou')

# add tp, fp, and fn
detect_df['tp@IoU'+str(iou_thres)] = np.where(((detect_df.predicted.notnull()) & (detect_df.iou >= iou_thres)) & detect_df.ground_truth.notnull(), '1', '0')
detect_df['fp@IoU'+str(iou_thres)] = np.where((detect_df.predicted.notnull()) & detect_df.ground_truth.isnull(), '1', '0')
detect_df['fn@IoU'+str(iou_thres)] = np.where(((detect_df.predicted.isnull()) | (detect_df.iou < iou_thres)) & detect_df.ground_truth.notnull(), '1', '0')
print(detect_df)

# save as .csv
detect_df.to_csv(path_or_buf=args.output_path)