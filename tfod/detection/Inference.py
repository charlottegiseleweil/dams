######
#
# Inference.py
#
# Inputs a TF inference graph(frozen model .pb, obtained with export_inference_graph.py),
# and a directory of images
#
# Runs inference, outputs bboxes, in a text file per image with format:
# {bbox1} {bbox2} {bbox3} {bbox4} {class(int)} {confidence}

# where bbox format depends on --bbox_format input. 
# Default is: y1x1y2x2_norm : Top, Left, Bottom, Right NORMALIZED (TFOD format)
#
# Usage: python inference.py
#          --modelFile {path_to_frozen_graph.pb} 
#          --imgDir {path_to_dir}
#          --outputDir {path_to_dir}
#          --bbox_format (Optional) {outputs bboxes format. Default is y1x1y2x2_pixel}
#          --confidenceThreshold (Optional) writes only bboxes above this threshold. Default to 0.05
#          --max_boxes_per_images (Optional) {int} Will write this # of bboxes per image. Default to 1
# Example usage: 
# python inference.py --modelFile ../../../../results/fasterRCNN_07-27_newimagery.pb --imgDir ../../../../data/one_img_sample --outputDir ../../../../results/sample_fasterRCNN_07-27_newimagery
#
# Dependencies:
# Needs run_tf_detector.py in the same directory
#
# (For visualization of results, see Inference.ipynb)
#
# Charlotte Weil, August 2019



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
	default='y1x1y2x2_norm',
	help='BBoxes format to write: x1y1x2y2_pixel, x1y1x2y2_norm, xywh_pixel or xywh_norm')
parser.add_argument('--confidenceThreshold', type=float,
    default='0.05',
    help='Will only write bboxes with score > confidenceThreshold')
parser.add_argument('--max_boxes_per_images', type=int,
    default='1',
    help='Will write this # of bboxes per image')

args = parser.parse_args()


## From run_tf_detector.py (CameraTraps)

def load_model(checkpoint):
    """
    Load a detection model (i.e., create a graph) from a .pb file
    """

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    return detection_graph


def generate_detections(detection_graph,images):
    """
    boxes,scores,classes,images = generate_detections(detection_graph,images)

    Run an already-loaded detector network on a set of images.

    [images] can be a list of numpy arrays or a list of filenames.  Non-list inputs will be
    wrapped into a list.

    Boxes are returned in relative coordinates as (top, left, bottom, right); 
    x,y origin is the upper-left.
    
    [boxes] will be returned as a numpy array of size nImages x nDetections x 4.
    
    [scores] and [classes] will each be returned as a numpy array of size nImages x nDetections.
    
    [images] is a set of numpy arrays corresponding to the input parameter [images], which may have
    have been either arrays or filenames.    
    """

    if not isinstance(images,list):
        images = [images]
    else:
        images = images.copy()

    print('Loading images...')
    startTime = time.time()
    
    # Load images if they're not already numpy arrays
    # iImage = 0; image = images[iImage]
    for iImage,image in enumerate(tqdm(images)):
        if isinstance(image,str):
            
            # Load the image as an nparray of size h,w,nChannels
            
            # There was a time when I was loading with PIL and switched to mpimg,
            # but I can't remember why, and converting to RGB is a very good reason
            # to load with PIL, since mpimg doesn't give any indication of color 
            # order, which basically breaks all .png files.
            #
            # So if you find a bug related to using PIL, update this comment
            # to indicate what it was, but also disable .png support.
            image = PIL.Image.open(image).convert("RGB"); image = np.array(image)
            # image = mpimg.imread(image)
            
            # This shouldn't be necessary when loading with PIL and converting to RGB
            nChannels = image.shape[2]
            if nChannels > 3:
                print('Warning: trimming channels from image')
                image = image[:,:,0:3]
            images[iImage] = image
        else:
            assert isinstance(image,np.ndarray)

    elapsed = time.time() - startTime
    print("Finished loading {} file(s) in {}".format(len(images),
          humanfriendly.format_timespan(elapsed)))    
    
    boxes = []
    scores = []
    classes = []
    
    nImages = len(images)

    print('Running detector...')    
    startTime = time.time()
    firstImageCompleteTime = None
    
    with detection_graph.as_default():
        
        with tf.Session(graph=detection_graph) as sess:
            
            for iImage,imageNP in tqdm(enumerate(images)): 
                
                imageNP_expanded = np.expand_dims(imageNP, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                box = detection_graph.get_tensor_by_name('detection_boxes:0')
                score = detection_graph.get_tensor_by_name('detection_scores:0')
                clss = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                
                # Actual detection
                (box, score, clss, num_detections) = sess.run(
                        [box, score, clss, num_detections],
                        feed_dict={image_tensor: imageNP_expanded})

                boxes.append(box)
                scores.append(score)
                classes.append(clss)
            
                if iImage == 0:
                    firstImageCompleteTime = time.time()
                    
            # ...for each image                
    
        # ...with tf.Session

    # ...with detection_graph.as_default()
    
    elapsed = time.time() - startTime
    if nImages == 1:
        print("Finished running detector in {}".format(humanfriendly.format_timespan(elapsed)))
    else:
        firstImageElapsed = firstImageCompleteTime - startTime
        remainingImagesElapsed = elapsed - firstImageElapsed
        remainingImagesTimePerImage = remainingImagesElapsed/(nImages-1)
        
        print("Finished running detector on {} images in {} ({} for the first image, {} for each subsequent image)".format(len(images),
              humanfriendly.format_timespan(elapsed),
              humanfriendly.format_timespan(firstImageElapsed),
              humanfriendly.format_timespan(remainingImagesTimePerImage)))
    
    nBoxes = len(boxes)
    
    # Currently "boxes" is a list of length nImages, where each element is shaped as
    #
    # 1,nDetections,4
    #
    # This implicitly banks on TF giving us back a fixed number of boxes, let's assert on this
    # to make sure this doesn't silently break in the future.
    nDetections = -1
    # iBox = 0; box = boxes[iBox]
    for iBox,box in enumerate(boxes):
        nDetectionsThisBox = box.shape[1]
        assert (nDetections == -1 or nDetectionsThisBox == nDetections), 'Detection count mismatch'
        nDetections = nDetectionsThisBox
        assert(box.shape[0] == 1)
    
    # "scores" is a length-nImages list of elements with size 1,nDetections
    assert(len(scores) == nImages)
    for(iScore,score) in enumerate(scores):
        assert score.shape[0] == 1
        assert score.shape[1] == nDetections
        
    # "classes" is a length-nImages list of elements with size 1,nDetections
    #
    # Still as floats, but really representing ints
    assert(len(classes) == nBoxes)
    for(iClass,c) in enumerate(classes):
        assert c.shape[0] == 1
        assert c.shape[1] == nDetections
            
    # Squeeze out the empty axis
    boxes = np.squeeze(np.array(boxes),axis=1)
    scores = np.squeeze(np.array(scores),axis=1)
    classes = np.squeeze(np.array(classes),axis=1).astype(int)
    
    # boxes is nImages x nDetections x 4
    assert(len(boxes.shape) == 3)
    assert(boxes.shape[0] == nImages)
    assert(boxes.shape[1] == nDetections)
    assert(boxes.shape[2] == 4)
    
    # scores and classes are both nImages x nDetections
    assert(len(scores.shape) == 2)
    assert(scores.shape[0] == nImages)
    assert(scores.shape[1] == nDetections)
    
    assert(len(classes.shape) == 2)
    assert(classes.shape[0] == nImages)
    assert(classes.shape[1] == nDetections)
    
    return boxes,scores,classes,images




def write_predicted_bb(boxes, scores, classes, inputFileNames, 
                       outputDir, confidenceThreshold=0.05,
                       output_format='txt_files',
                       max_boxes_per_images=1,
                       bbox_format='y1x1y2x2_norm'):
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


        if iImage%1==0:
            print('Wrote bbox text files for ',iImage,' images')

        inputFileName = inputFileNames[iImage]
        imageName = inputFileName.rsplit('/',1)[-1][:-4]
        outputFile = os.path.join(outputDir,imageName+'.txt')

        image = mpimg.imread(inputFileName)
        iBox = 0; box = boxes[iImage][iBox]

        for iBox,box in enumerate(boxes[iImage]):

            score = scores[iImage][iBox]
            if score < confidenceThreshold:
                continue

            if iBox >= max_boxes_per_images:
                break

            if bbox_format=='y1x1y2x2_pixel':
                s = image.shape; imageHeight = s[0]; imageWidth = s[1]
                string_to_write = str(box[0]*s[0])+' '+str(box[1]*s[1])+' '+str(box[2]*s[0])+' '+str(box[3]*s[1])+' '+str(detected_class)+' '+str(score)

            elif bbox_format=='y1x1y2x2_norm':
                string_to_write = str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+' '+str(detected_class)+' '+str(score)

            elif bbox_format=='xywh_norm':
                print('xywh_norm format todo -- Let Charlie know if you need this!')

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

                print('xywh_pixel format todo -- Let Charlie know if you need this!')

            else:
                print(bbox_format,' format not supported yet -- Let Charlie know if you need this!')

            
            file = open(outputFile,'w')
            file.write(string_to_write)
            file.close()




def load_and_run_detector_to_bboxes(modelFile,
                                    imgDir,
                                    outputDir,
                                    confidenceThreshold=0.05,
                                    max_boxes_per_images=1,
                                    bbox_format='y1x1y2x2_norm'):

    """
    Wrappper to load and run detector, then write bboxes.

    """
    

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

