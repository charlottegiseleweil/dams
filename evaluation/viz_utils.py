######
#
# Evaluation_Utils.py
#
# Functions needed for Visualization, done in Evaluation notebook.
# Work on a results dataframe (with cols)
#
# Charlotte Weil, August 2019



# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shapely.geometry import box
import matplotlib.ticker as ticker

from sklearn import metrics
import logging
logging.getLogger().setLevel(logging.INFO)

# State variables
cols = ['img_id','gt_bbox','gt_format','pred_bbox','pred_format','confidence','gt_size','iou']


# Useful Bbox format Conversion Functions

def parse_bbox_str_to_list(bbox_str):
    '''Parse [a b c d] to [a,b,c,d]'''
    logging.debug(bbox_str)
    bbox_without_brackets = bbox_str.replace('[ ','').replace('[','').replace(']','').replace('  ',' ').replace('  ',' ')
    vals = bbox_without_brackets.split(' ')
    logging.debug(vals)
    bbox_list = [float(vals[0]),float(vals[1]),float(vals[2]),float(vals[3])]
    return bbox_list

def convert_bbox_pixel_to_norm(bbox_list,imgSize=419):
    '''Convert bbox_list from _pixel to normalized'''
    return [i/imgSize for i in bbox_list]

def convert_bbox_xywh_to_x1y1x2y2(bbox_list):
    x_min, y_min, width_of_box, height_of_box = bbox_list
    x_max, y_max = x_min + width_of_box, y_min + height_of_box
    return [x_min, y_min, x_max, y_max]



# Visualization function


def visualize_bboxes(df, images_dir,
                     max_images=100,
                    BIG_IMAGES=False,
                    SHOW_CONFIDENCE_VALUES=True,
                    NO_BOXES=False):
    
    if df.index.name != 'img_id':
        df = df.set_index('img_id')

    fig = plt.figure(figsize=(12,240))
    
    #randlist = random.sample(range(0, len(os.listdir(images_dir))), how_many)

    
    k=0
    for img_id in df.index:
        k+=1
        
        #  Image
        img_fn= img_id+'.png'
        img_fp = os.path.join(images_dir, img_fn)
        img = mpimg.imread(img_fp)
        s = img.shape; imageHeight = s[0]; imageWidth = s[1]
        logging.debug(img_id)
         
        # GT_bbox
        if pd.notnull(df.loc[img_id].gt_bbox) and NO_BOXES!=True:
            gt_coords = parse_bbox_str_to_list(df.loc[img_id].gt_bbox)
            if df.iloc[0].gt_format == 'xywh_pix':
                gt_bbox = box(gt_coords[0], gt_coords[1], gt_coords[2], gt_coords[3])
            elif df.iloc[0].gt_format == 'y1x1y2x2_pix':
                gt_bbox = box(gt_coords[1], gt_coords[2], gt_coords[3], gt_coords[0])
            x_gt,y_gt = gt_bbox.exterior.xy
            
        # Predicted_bbox
        if pd.notnull(df.loc[img_id].pred_bbox) and NO_BOXES!=True:
            pred_coords = parse_bbox_str_to_list(df.loc[img_id].pred_bbox)
            if df.iloc[0].pred_format == 'xywh_pix':
                pred_bbox = box(pred_coords[0], pred_coords[1], pred_coords[2], pred_coords[3])
            elif df.iloc[0].pred_format == 'y1x1y2x2_pix':
                pred_bbox = box(pred_coords[1], pred_coords[2], pred_coords[3], pred_coords[0])
            x_pr,y_pr = pred_bbox.exterior.xy
            # Confidence & IoU
            img_conf = df.loc[img_id].confidence
            img_iou = str('%.2f'%(float(df.loc[img_id].iou))) 
 
            
        # Plot
        ax = fig.add_subplot(20,1,k)
        if NO_BOXES!=True:
            ax.set_title(img_id)# + '    confidence: ' + img_conf + '    iou: ' + img_iou)
        ax.imshow(img)
        
        try: # If there are GT bboxes
            ax.plot(x_gt,y_gt,color='lightblue',linewidth=4)
        except NameError:
            pass
        
        try: # If there are predicted bboxes
            ax.plot(x_pr,y_pr,color='orange',linewidth=4)
            
            if SHOW_CONFIDENCE_VALUES == True:
                 # origin is top left
                iLeft = pred_coords[1]   #(origin x)
                iBottom = pred_coords[0] #(origin y)
                pLabel = '({:.2f})'.format(img_conf)
                ax.text(iLeft+5,iBottom+5,pLabel,
                        color='orange',fontsize=12,
                        verticalalignment='top')
        except NameError:
            pass

        # This is magic goop that removes whitespace around image plots (sort of)        
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        if BIG_IMAGES == True:
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, 
                            wspace = 0)
            plt.margins(0,0)
            ax.xaxis.set_major_locator(ticker.NullLocator())
            ax.yaxis.set_major_locator(ticker.NullLocator())
            ax.axis('tight')
            ax.set(xlim=[0,imageWidth],ylim=[imageHeight,0],aspect=1)
            plt.axis('off')  
            
