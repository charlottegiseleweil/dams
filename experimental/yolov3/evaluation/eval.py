# imports
import argparse
import os
import pandas as pd
import numpy as np
import json 
from copy import deepcopy

# inputs
parser = argparse.ArgumentParser(description='Custom COCO metrics')
parser.add_argument('--images', type=str, default='../../../../outputs/detection-results/images', help='directory of images output from detect.py')
parser.add_argument('--predicted_bboxes', type=str, default='../../../../outputs/detection-results/pred', help='directory of .txt files of predicted bounding boxes')
parser.add_argument('--ground_truth_bboxes', type=str, default='../../../../outputs/detection-results/truth')
parser.add_argument('--iou_thres', type=float, default=0.5, help='IoU threshold for mAP calculation')
parser.add_argument('--size_thres', type=str, default='32,96', help='threshold for size categories (input for COCO categories would be: 32,96)')
# use tuple
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

# combine pred bbox and 
def parse_pred_bbox (label_fp):
    """Obtain x_min, y_min, x_max, and y_max of bounding box from txt file
    Args:
        label_fp (str): filepath to bounding box .txt file in detect.py output format
    Returns:
        coords (list of list)
    """
    with open(label_fp, 'r') as label:
        line = str(label.readline())
        vals = line.split(' ')
        x_min = int(vals[0])
        y_min = int(vals[1])
        x_max = int(vals[2])
        y_max = int(vals[3])
    coords = [[x_min, y_min, x_max, y_max]]
    return coords

def parse_pred_box_conf (label_fp):
    """Obtain confidence score of bounding box from .txt file
    Args:
        label_fp (str): filepath to bounding box .txt file in detect.py output format
    Returns:
        coords (list of list)
    """
    with open(label_fp, 'r') as label:
        line = str(label.readline())
        vals = line.split(' ')
        conf = float(vals[5])
        conf = [float('%.4f'%(conf))]
    return conf

def parse_gt_bbox (label_fp):
    """Obtain confidence score of bounding box from .txt file
    Args:
        label_fp (str): filepath to bounding box .txt file in darknet input format
    Returns:
        coords (list of list)
    """
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
    coords = [[x_min, y_min, x_max, y_max]]
    return coords

# 
def calc_IoU (ground_truth, predicted):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    """
    # gather gt and pred coords
    x1_t, y1_t, x2_t, y2_t = ground_truth[0]
    x1_p, y1_p, x2_p, y2_p = predicted[0]
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

# function to calculate IoU from bboxes in np.array format
def calc_IoUa (ground_truth, predicted):
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

def get_model_conf_map (pred_boxes):
    """Creates a dictionary of from model_scores to image ids.
    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    model_conf_map = {}
    for img_id, val in pred_boxes.items():
        for conf in val['conf']:
            if conf not in model_conf_map.keys():
                model_conf_map[conf] = [img_id]
            else:
                model_conf_map[conf].append(img_id)
    return model_conf_map

def get_single_image_results (gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}   
    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_IoUa(gt_box, pred_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

def calc_precision_recall (img_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        if res is None:
            precision = 0.0
            recall = 0.0
        else:
            true_pos += res['true_pos']
            false_pos += res['false_pos']
            false_neg += res['false_neg']
    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0
    return (precision, recall)

def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr):
    """Calculates average precision at given IoU threshold.

    error -> should be dict 

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: avg precision as well as summary info about the PR curve
        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    model_conf_map = get_model_conf_map(pred_boxes)
    sorted_model_conf = sorted(model_conf_map.keys())
    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for img_id in pred_boxes.keys():
        arg_sort = np.argsort(pred_boxes[img_id]['conf'])
        pred_boxes[img_id]['conf'] = np.array(pred_boxes[img_id]['conf'])[arg_sort].tolist()
        pred_boxes[img_id]['predicted'] = np.array(pred_boxes[img_id]['predicted'])[arg_sort].tolist()
    pred_boxes_pruned = deepcopy(pred_boxes)
    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    # Loop over model conf thresholds and calculate precision, recall
    for ithr, model_conf_thr in enumerate(sorted_model_conf[:-1]):
        # On first iteration, define img_results for the first time:
        img_ids = gt_boxes.keys() if ithr == 0 else model_conf_map[model_conf_thr]
        for img_id in img_ids:
            if img_id not in pred_boxes_pruned.keys():
                img_results[img_id] = {'true_pos': 0, 'false_pos': 0, 'false_neg': 1}
            else:
                gt_boxes_img = gt_boxes[img_id]
                box_conf = pred_boxes_pruned[img_id]['conf']
                start_idx = 0
                for conf in box_conf:
                    if conf <= model_conf_thr:
                        pred_boxes_pruned[img_id]
                        start_idx += 1
                    else:
                        break
                pred_boxes_pruned[img_id]['conf'] = pred_boxes_pruned[img_id]['conf'][start_idx:]
                pred_boxes_pruned[img_id]['predicted'] = pred_boxes_pruned[img_id]['predicted'][start_idx:]
                img_results[img_id] = get_single_image_results(gt_boxes_img, pred_boxes_pruned[img_id]['predicted'], iou_thr)
        prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_conf_thr)
    print('precisions:',precisions)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)
    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'model_thrs': model_thrs}

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
detect_df['conf'] = detect_df['image'].map(pd.Series(conf_dict))

# add ground_truth bbox size to dataframe
detect_df['size'] = detect_df.apply(
    lambda row: int((row.ground_truth[0][2] - row.ground_truth[0][0]) * (row.ground_truth[0][3] - row.ground_truth[0][1])), 
    axis=1
)

# add IoU to dataframe
# not a dam needs to be accounted for 
iou_dict = {}
for fn in predicted_dict:
    iou = calc_IoU(ground_truth_dict[fn], predicted_dict[fn])
    iou_dict[fn] = iou
detect_df['iou'] = detect_df['image'].map(pd.Series(iou_dict))

# create input jsons to calculate mAP
# description
detect_df = detect_df.set_index('image')
pred_df_a = detect_df[detect_df['predicted'].notnull()]
pred_df_b = pred_df_a[['predicted', 'conf']]
pred_json_str = pred_df_b.to_json(orient='index')
gt_r = json.dumps(ground_truth_dict)
gt_json = json.loads(gt_r)
pred_json = json.loads(pred_json_str)
with open('gt_json', 'w') as gt_json:
    json.dump(ground_truth_dict, gt_json)
with open('pred_json', 'w') as pred_json:
    json.dump(pred_json_str, pred_json)

# calculate average precision
data = get_avg_precision_at_iou(gt_json, pred_json, iou_thres)
print('avg precision: {:.4f}'.format(data['avg_prec']))