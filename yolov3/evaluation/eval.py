# imports
import argparse
import os
import pandas as pd
import numpy as np

# inputs
parser = argparse.ArgumentParser(description='Custom COCO metrics')
parser.add_argument('--images', type=str, default='../../../../outputs/detection-results/images', help='directory of images output from detect.py')
parser.add_argument('--predicted_bboxes', type=str, default='../../../../outputs/detection-results/pred', help='directory of .txt files of predicted bounding boxes')
parser.add_argument('--ground_truth_bboxes', type=str, default='../../../../outputs/detection-results/truth')
parser.add_argument('--iou_thres', type=float, default=0.5, help='IoU threshold for mAP calculation')
parser.add_argument('--size_thres', type=str, default='32,96', help='threshold for size categories (input for COCO categories would be: 32,96)')
parser.add_argument('--image_size', type=int, default=419, help='length of image edge')
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
size_thres = args.size_thres
lim_small = int(size_thres.split(',')[0]) ** 2
lim_medium = int(size_thres.split(',')[1]) ** 2
image_size = args.image_size

# function to parse predicted bounding box txt files for min/max values
def parse_pred_bbox (label_fp):
	with open(label_fp, 'r') as label:
		line = str(label.readline())
		vals = line.split(' ')
		x_min = int(vals[0])
		y_min = int(vals[1])
		x_max = int(vals[2])
		y_max = int(vals[3])
	coords = np.array([x_min, y_min, x_max, y_max])
	return coords

# function to parse predicted bounding box txt files for confidence level
def parse_pred_box_conf (label_fp):
	with open(label_fp, 'r') as label:
		line = str(label.readline())
		vals = line.split(' ')
		conf = float(vals[5])
		conf = '%.4f'%(conf)
	return conf

# function to parse ground truth bounding box txt files for min/max values
def parse_gt_bbox (label_fp):
	with open(label_fp, 'r') as label_txt:
		line = label_txt.readline()
		norm_x = float(line[2:10])
		norm_y = float(line[11:19])
		norm_w = float(line[20:28])
		norm_h = float(line[29:])
		x_min = int((norm_x * image_size) - ((norm_w * image_size) / 2))
		y_min = int((norm_y * image_size) - ((norm_h * image_size) / 2))
		x_max = int((norm_x * image_size) + ((norm_w * image_size) / 2))
		y_max = int((norm_y * image_size) + ((norm_h * image_size) / 2))
	coords = np.array([x_min, y_min, x_max, y_max])
	return coords

# function to calculate IoU from bboxes in np.array format
def calc_IoU (ground_truth, predicted):
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

# function to calculate precision and recall for input dataframe (with same columns as detect_df)
def calc_precision_and_recall (df):
	pred_df = df[df['predicted'].notnull()]
	gt_df = df[df['ground_truth'].notnull()]
	true_pos = len(pred_df[pred_df['ground_truth'].notnull()])
	false_pos = len(pred_df[pred_df['ground_truth'].isnull()])
	false_neg = len(gt_df[gt_df['predicted'].isnull()])
	precision = float('%.4f'%(true_pos / (true_pos + false_pos)))
	recall = float('%.4f'%(true_pos / (true_pos + false_neg)))
	p_r_dict = {'precision':precision, 'recall':recall}
	return p_r_dict

# create dictionaries of ground_truth and predicted bounding boxes
ground_truth_dict = {}
for fn in gt_bbox_fn:
	bbox = parse_gt_bbox(os.path.join(gt_bbox_dir, fn))
	ground_truth_dict[fn.replace('.txt', '.png')] = bbox
predicted_dict = {}
for fn in pred_bbox_fn:
	bbox = parse_pred_bbox(os.path.join(pred_bbox_dir, fn))
	predicted_dict[fn.replace('.txt', '.png')] = bbox
conf_dict = {}
for fn in pred_bbox_fn:
	conf = parse_pred_box_conf(os.path.join(pred_bbox_dir, fn))
	conf_dict[fn.replace('.txt', '.png')] = conf

# build dataframe of image, ground_truth bboxes, predicted bboxes, and confidence
data = {'image' : image_fn}
detect_df = pd.DataFrame(data, columns=['image'])
detect_df['ground_truth'] = detect_df['image'].map(pd.Series(ground_truth_dict))
detect_df['predicted'] = detect_df['image'].map(pd.Series(predicted_dict))
detect_df['confidence'] = detect_df['image'].map(pd.Series(conf_dict))

# add ground_truth bbox size to dataframe
detect_df['size'] = detect_df.apply(
	lambda row: int((row.ground_truth[2] - row.ground_truth[0]) * (row.ground_truth[3] - row.ground_truth[1])), 
	axis=1
)

# add IoU to dataframe
iou_dict = {}
for fn in predicted_dict:
	iou = calc_IoU(ground_truth_dict[fn], predicted_dict[fn])
	iou_dict[fn] = iou
detect_df['iou'] = detect_df['image'].map(pd.Series(iou_dict))
#detect_df = detect_df.fillna(0)
print(detect_df)

# split dataframe by size threholds
detect_small_df = detect_df[detect_df['size'] <= lim_small]
detect_medium_df = detect_df[detect_df['size'] <= lim_medium]
detect_large_df = detect_df[detect_df['size'] >= lim_medium]

# split dataframe by IoU threshold
detect_iou_df = detect_df[detect_df['iou'] >= iou_thres]

# calculate precision and recall
p_r = calc_precision_and_recall(detect_df)

print(p_r)