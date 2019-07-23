#!/usr/bin/env python
# coding: utf-8

# * Step 1) Exports a trained model into a frozen graph
# * Step 2) Runs inference on sample images
# 
# */!\ Can't have 2 notebooks with tf.Sessions in parrallel !*
# 
# Run from environment with requirements:
# * TFOD All things
# * humanfriendly
# * *e.g use docker epic_cray in Cobbdam.*

# In[67]:


import os
import tqdm
import numpy as np

path_to_root = '../../../..'
path_to_outputs = os.path.join(path_to_root,'outputs/fasterRCNN_07_02_img_resize')


# # Step 1) Export graph (.pbtxt to .pb)

# ### Option a) Freeze graph w/ tf.python.tool.freeze_graph()
# Issue: Hard to determine output_node_names

# In[ ]:


from tensorflow.python.tools import freeze_graph


# In[ ]:


pbtxt_path = os.path.join(path_to_outputs, 'graph.pbtxt')
ckpt_path = os.path.join(path_to_outputs, 'model.ckpt-300000')
frozen_graph_path = os.path.join(path_to_outputs, "frozen_graph_MergeSummary.pb")


# In[ ]:


freeze_graph.freeze_graph(input_graph=pbtxt_path,
                          input_saver='',input_binary=False,
                          input_checkpoint=ckpt_path,
                          output_node_names='Merge/MergeSummary', 
                          #output_node_names = 'group_deps', 
                          #output_node_names ? Found the last node name in pbtxt_file: 'Merge/MergeSummary'
                          # Or a random one in TensorBoard: 'group_deps', 
                          
                          restore_op_name='save/restore_all',
                          filename_tensor_name='save/Const:0',
                          output_graph=frozen_graph_path,
                          clear_devices=True,initializer_nodes='')


# ### Option b) With export_inference_graph.py
# 
# Dst tensors errors occur if many things running. Shutdown kernel and re-launch jupyter.

# In[ ]:


os.mkdir(os.path.join(path_to_root,'outputs/fasterRCNN_07_02_img_resize/export_inference'))
os.path.join(path_to_root,'outputs/fasterRCNN_07_02_img_resize/export_inference')


# In[5]:


get_ipython().system(' python export_inference_graph.py     --input_type image_tensor     --pipeline_config_path ../../../../repos/dams/tfod/training/configs/07_02_imgsize_faster_rcnn_resnet50_coco.config     --trained_checkpoint_prefix ../../../../outputs/fasterRCNN_07_02_img_resize/model.ckpt-300000     --output_directory ../../../../outputs/fasterRCNN_07_02_img_resize/export_inference     --write_inference_graph True')


# # Inference

# In[32]:


import argparse
import glob
import os
import sys
import time

import PIL
import humanfriendly

## IMPORTANT to avoid $DISPLAY problems:
import matplotlib
matplotlib.use('Agg')
get_ipython().run_line_magic('matplotlib', 'inline')


import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from tqdm import tqdm


# In[20]:


from IPython.display import Image
from run_tf_detector import *

DEFAULT_CONFIDENCE_THRESHOLD = 0.85


# In[74]:


how_many_sample_images = 40

frozen_graph_path = os.path.join(path_to_outputs, "export_inference/frozen_inference_graph.pb")

sample_images = []
for filename in os.listdir(os.path.join(path_to_root,'data/Sample_imagery_6-7'))[:how_many_sample_images]:
    sample_images.append(os.path.join(path_to_root,'data/Sample_imagery_6-7',filename))
    
outputDir = os.path.join(path_to_outputs, 'infered_sample40') 
if not os.path.exists(outputDir):
    os.mkdir(outputDir)


# In[78]:


load_and_run_detector(modelFile=frozen_graph_path,
                      imageFileNames= sample_images,
                      outputDir=outputDir,
                      confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD)


# In[79]:


for filename in os.listdir(outputDir):
    i = os.path.join(outputDir, filename)
    print(filename)
    display(Image(i))


# ### For humans to play to see if they are better than my algo!

# In[82]:


sample_images

for filename in sample_images:
    print(filename)
    display(Image(filename))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Experiments: Steps by steps from run_tf_detector (instead of using wrapper fct load_and_run_detector() )

# In[8]:


## The base pre-trained model : faster_rcnn_resnet50_coco

frozen_graph_path = os.path.join(path_to_root, "pre-trained-models/faster_rcnn_resnet50_coco/frozen_inference_graph.pb")
frozen_graph_path


# In[40]:


sample_img_path = os.path.join(path_to_root,'data/Sample_imagery_6-7/116421_clipped.png')
sample_img_path


# In[56]:


detection_graph = load_model(frozen_graph_path)
detection_graph


# In[61]:


boxes,scores,classes,images = generate_detections(detection_graph,sample_img_path)


# In[62]:


outputDir = os.path.join(path_to_outputs, 'infered_sample') 
outputFileName = os.path.join(outputDir, 'test_sample_img2.png') 


# In[63]:


render_bounding_boxes(boxes,
                      scores,
                      classes,
                      inputFileNames=[sample_img_path],
                      outputFileNames=[outputFileName],
                      confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD,
                      linewidth=DEFAULT_LINE_WIDTH)


# In[64]:


from IPython.display import Image
Image(filename=outputFileName) 


# ### Now with OUR model 

# In[22]:


frozen_graph_path = os.path.join(path_to_outputs, "export_inference/frozen_inference_graph.pb")
frozen_graph_path


# In[23]:


detection_graph = load_model(frozen_graph_path)


# In[41]:


boxes,scores,classes,images = generate_detections(detection_graph,sample_img_path)


# In[42]:


outputDir = os.path.join(path_to_outputs, 'infered_sample') 
outputFileName = os.path.join(outputDir, 'test_sample_img3_my.png') 


# In[48]:


render_bounding_boxes(boxes,
                      scores,
                      classes,
                      inputFileNames=[sample_img_path],
                      outputFileNames=[outputFileName],
                      confidenceThreshold=DEFAULT_CONFIDENCE_THRESHOLD,
                      linewidth=DEFAULT_LINE_WIDTH)


# In[49]:


from IPython.display import Image
Image(filename=outputFileName) 


# In[ ]:




