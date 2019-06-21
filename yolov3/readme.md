
## Installation Procedure

Clone the darknet repository:

```
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
```

Download convolutional weights pretrained on ImageNet:

```
wget https://pjreddie.com/media/files/darknet53.conv.74
```

## Prepare YOLO-ready data

#### Run `prepare_darknet_training_data.ipynb`
+ Inputs:
    + zip of all dam images: `dam.zip`
    + zip of all not dam images: `not_a_dam.zip`
+ Outputs:
    + `/images` directory with `/train` and `/test` subdirectories
    + `/labels` directory with `/train` and `/test` subdirectories
        + each label contains the mean x value, mean y value, width, and height of the bounding box in the corresponding image file
        + all numbers are normalized between 0 and 1
    + `dams.names`
        + contains class names
    + `dams.data`
        + contains directory information for darknet
    + `test_images.txt`
    + `train_images.txt`
        
#### Modify `yolov3.cfg`

+ Create of a copy `/darknet/cfg/yolov3.cfg` in your directory
+ Modify `yolov3.cfg` file to work on one class:
    + Line 3, set `batch=16`
    + Line 4, set `subdivisions=16`
    + Line 603, set `filters=18`, filters = (number_of_classes + 5) * 3 
    + Line 610, set `classes=1`, we have one class (dams)
    + Line 689, set `filters=18`, filters = (number_of_classes + 5) * 3 
    + Line 696, set `classes=1`, we have one class (dams)
    + Line 776, set `filters=18`, filters = (number_of_classes + 5) * 3 
    + Line 783, set `classes=1`, we have one class (dams)
    + we set 18 filters because each cell in YOLOv3 predicts 3 bounding boxes. Each bounding box has 5 + number_of_classes attributes (dimenions, objectness score, class confidence)
+ Modify name of cfg file to keep track of edits
        
## Training Procedure

#### Required files and directories:
+ Classes
   + `dams.names'`
   + contains class names, there should be only one, dam
+ Directory Paths
   + `dams.data`
   + contains directory paths for darknet
   + this includes paths for the `/images`, `/labels`, `train.txt`, `test.txt`, and `backup`
+ Modified `dams.cfg` (see above)
+ Image directory
   + `/images`, path should be in `dams.data`
+ Labels directory
   + `/labels`, path should be in `dams.data`
+ pretrained weights: `darknet53.conv.74`

#### Train model

Run `./darknet detector train /path/to/dams.data /path/to/dams.cfg /path/to/weights`
+ Example: `./darknet detector train yolov3/dams.data yolov3/dams.cfg darknet53.conv.74`
