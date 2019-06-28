## Installation Procedure

+ Python 3.7 or later is required
+ Create and activate new virtual environment:
  + `virtualenv -p python3.7 ultralytics_yolov3_env`
  + `source ultralytics_yolov3_env/bin/activate`
+ Clone the Ultralytics-YOLOv3 repository:
  + `git clone https://github.com/ultralytics/yolov3`
+ Install dependencies
  + `pip install -U -r requirements.txt`
  + Make sure you are within Ultralytics-YOLOv3 repo and using `ultralytics_yolov3_env` virtual environment
  
## Prepare Inputs

#### Run `Make_darknet_inputs.ipynb`
+ Inputs:
    + directory of dam images
    + directory of not_a_dam images
+ Outputs:
    + `/images` directory with `/train`, `/test`, `validation`, and `southaf` subdirectories
    + `/labels` directory with `/train`, `/test`, `validation`, and `southaf` subdirectories
        + each label contains the mean x value, mean y value, width, and height of the bounding box in the corresponding image file
        + all numbers are normalized between 0 and 1
    + `.txt` files with absoulte paths to images and labels of the four sets:
        + `test_images_filepaths_abs.txt`
        + `valid_images_filepaths_abs.txt`
        + `test_images_filepaths_abs.txt`
        + `southaf_images_filepaths_abs.txt`
    + `[run-date].names`
        + contains class names
    + `[run_date].data`
        + contains directory information for darknet

#### Modify config file
+ Create a copy of `/ultralytics/yolov3/cfg/yolov3.cfg`
+ Rename copy `[run-date].cfg`
+ Modify `[run-date].cfg` to allow for training with one class (as detailed in https://github.com/WyattAutomation/Train-YOLOv3-with-OpenImagesV4#cfg-file):
    + Line 7, set `batch=16`
    + Line 603, set `filters=18`, filters = (number_of_classes + 5) * 3 
    + Line 610, set `classes=1`, we have one class (dams)
    + Line 689, set `filters=18`, filters = (number_of_classes + 5) * 3 
    + Line 696, set `classes=1`, we have one class (dams)
    + Line 776, set `filters=18`, filters = (number_of_classes + 5) * 3 
    + Line 783, set `classes=1`, we have one class (dams)
    
## Training Procedure

#### Organize directories
+ In `/dams` repo, move the following files to `/yolov3/cfg`:
    + `[run-date].cfg`
    + `[run-date].names`
    + `[run_date].data`
+ Move the following files into `/yolov3/cfg/filepaths_txt` adjusting the absolute paths in `[run-date].data` as necessary
    + `test_images_filepaths_abs.txt`
    + `valid_images_filepaths_abs.txt`
    + `test_images_filepaths_abs.txt`
    + `southaf_images_filepaths_abs.txt`
    
#### Train model
+ Important Information
    + Ultralytics automatically applies `darknet53.conv.74` weights pre-trained on imageNet
    + mAP and other evaluation metrics are calculated using `pycocotools`
    + Navigate to Ultralytics-YOLOv3 cloned repo to train
+ Run: 
    + Create a new tmux session
        + `tmux new -s [session_name]`
    + Run: `python3 train.py --cfg ../dams/yolov3/cfg/[run-date].cfg --data-cfg ../dams/yolov3/cfg/[run-date].data`
+ Outputs:
    + Navigate to `/weights` to view outputs
    + Every 10 epochs, `backup[x]0.pt` is output
    + Weights with the highest mAP is output under `best.pt`
    + Weights from the most recent epoch is output under `latest.pt`
    + Navigate to the clone ultralytics repo to view `results.txt`
        + Second column from right contains mAP, as defined in `pycocotools`
        + Third column from right contains recall, as defined in `pycocotools`
        + Fourth column from right contains precision, as defined in `pycocotools`
    + <b>IMPORTANT<b>: Move all outputs to `~/charlie/outputs/[run_name]/` to avoid being overwritten during the next run



