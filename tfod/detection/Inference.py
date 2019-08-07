######
#
# Inference.py
#
# Inputs a TF inference graph(frozen model .pb, obtained with export_inference_graph.py),
# and a directory of images
#
# Runs inference, outputs bboxes, in a text file per image with format:
# {x_min} {y_min} {x_max} {y_max} {class(int)} {confidence}
#
# (For visualization of results, see Inference.ipynb)
#
# Charlotte Weil, August 2019
# 
#
# TODOs/WIP:
# bbox_formats not all supported yet.


# imports

import argparse
import os
import pandas as pd
import numpy as np

import argparse
import glob
import os
import sys
import time

import PIL
import humanfriendly

## IMPORTANT to avoid $DISPLAY problems:
import matplotlib
matplotlib.use('Agg') #TkAgg


import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Inputs

parser = argparse.ArgumentParser()

parser.add_argument('--modelFile', type=str, 
	help='TF Inference graph : frozen_model.pb, obtained with export_inference_graph.py')
parser.add_argument('--imgDir', type=str,
	help='directory of images to run inference on')
parser.add_argument('--outputDir', type=str,
	help='the directory where the txt files (one xxx.txt per xxx.png image) will be written')

parser.add_argument('--bbox_format', type=str,
	default='y1x1y2x2',
	help='BBoxes format to write: x1y1x2y2_pixel, x1y1x2y2_norm, xywh_pixel or xywh_norm')
parser.add_argument('--confidenceThreshold', type=float,
	default='0.01',
	help='Will only write bboxes with score > confidenceThreshold')
parser.add_argument('--max_boxes_per_images', type=int,
	default='1',
	help='Will write this # of bboxes per image')

args = parser.parse_args()



## Step 1/



def write_predicted_bb(boxes, scores, classes, inputFileNames, 
                       outputDir, confidenceThreshold=0.05,
                       output_format='txt_files',
                       max_boxes_per_images=1,
                       bbox_format='y1x1y2x2'):
    """
    Outputs bbox: image_name.png, predicted_bbox, confidence
    
    Writes json in format: 
    {"image_name.png": {"predicted": [[<x_min>,<y_min>,<x_max>,<y_max>]], "confidence"[[<confidence>]]}}
    
    
    [boxes] and [scores] should be in the format returned by generate_detections, 
    specifically [top, left, bottom, right] in normalized units, where the
    origin is the upper-left.    
    
    "classes" is currently unused, it's a placeholder for adding text annotations
    later.
    
    Args:
     - max_boxes_per_images : max number of bboxes per images. 
     - confidenceThreshold : Will return only bboxes with confidence > confidenceThreshold
     
     - bbox_format: 'xywh_norm' for bbox normalized to image size: x, y, width, height
                     'xywh_pixel' for bbox in pixels: x, y, width, height
                     'x1y1x2y2_norm' ...
                     
                     
     - output_format: txt_files leads to a txt files per image, named inputFileName.txt
                      json leads to one json file : 
                          {"image_name.png": {"predicted": [[<x_min>,<y_min>,<x_max>,<y_max>]], "confidence"[[<confidence>]]}}
    """

    detected_class = 0
    
    nImages = len(inputFileNames)
    iImage = 0

    for iImage in range(0,nImages):
        
        
        if iImage%1000==0:
            print('Wrote bbox text files for ',iImage,' images')
            
        inputFileName = inputFileNames[iImage]
        imageName = inputFileName.rsplit('/',1)[-1][:-4]
        outputFile = os.path.join(outputDir,imageName+'.txt')
        
        image = mpimg.imread(inputFileName)
        iBox = 0; box = boxes[iImage][iBox]
        
        #dpi = 100
        #s = image.shape; imageHeight = s[0]; imageWidth = s[1]
        #figsize = imageWidth / float(dpi), imageHeight / float(dpi)
        
        for iBox,box in enumerate(boxes[iImage]):

            score = scores[iImage][iBox]
            if score < confidenceThreshold:
                continue
    
            if iBox >= max_boxes_per_images:
                break

            if bbox_format=='y1x1y2x2':
                string_to_write = str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+' '+str(detected_class)+' '+str(score)
                
                
                file = open(outputFile,'w') 
                file.write(string_to_write)
                file.close() 
                
            elif bbox_format=='xywh_norm':  
                print('xywh_norm format todo')#TODO
            
            elif bbox_format=='xywh_pixel':
                topRel = box[0]
                leftRel = box[1]
                bottomRel = box[2]
                rightRel = box[3]

                s = image.shape; imageHeight = s[0]; imageWidth = s[1]
                x = leftRel * imageWidth
                y = topRel * imageHeight
                w = (rightRel-leftRel) * imageWidth
                h = (bottomRel-topRel) * imageHeight

                print()#TODO
            
            elif bbox_format=='minmax':
                #...
                print('minmax format not supported yet')
            
            
def load_and_run_detector_to_bboxes(modelFile,
                                    imgDir,
                                    outputDir,
                                    confidenceThreshold=0.05,
                                    max_boxes_per_images=1,
                                    bbox_format='y1x1y2x2'):
    
    imageFileNames = [os.path.join(imgDir,f) for f in os.listdir(imgDir) if f.endswith('.png')]
    
    if len(imageFileNames) == 0:        
        print('Warning: no files available')
        return
        
    # Load and run detector on target images
    print('Loading model...')
    startTime = time.time()
    detection_graph = load_model(modelFile)
    elapsed = time.time() - startTime
    print("Loaded model in {}".format(humanfriendly.format_timespan(elapsed)))
    
    boxes,scores,classes,images = generate_detections(detection_graph,imageFileNames)
    
    assert len(boxes) == len(imageFileNames)
    
    
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    
    print('Writing predicted bboxes files ...')
    write_predicted_bb(boxes=boxes, scores=scores, classes=classes,
                       inputFileNames=imageFileNames, 
                       outputDir=outputDir, confidenceThreshold=confidenceThreshold,
                       output_format='txt_files',
                       max_boxes_per_images=max_boxes_per_images,
                       bbox_format=bbox_format)
    
    

if __name__ == '__main__':

    load_and_run_detector_to_bboxes(modelFile=args.modelFile,
                                    imgDir=args.imgDir,
                                    outputDir=args.outputDir,
                                    confidenceThreshold=args.confidenceThreshold,
                                    max_boxes_per_images=args.max_boxes_per_images,
                                    bbox_format=args.bbox_format)
              
