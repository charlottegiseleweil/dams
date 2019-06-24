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

#### Run `Make_Darknet_inputs.ipynb`
+ Inputs:
    + directory of dam images
    + directory of not_a_dam images
+ Outputs:
    + `/images` directory with `/train`, `/test`, `validation`, and `southaf` subdirectories
    + `/labels` directory with `/train`, `/test`, `validation`, and `southaf` subdirectories
        + each label contains the mean x value, mean y value, width, and height of the bounding box in the corresponding image file
        + all numbers are normalized between 0 and 1
    + `test_images_filepaths.txt`
    + `valid_images_filepaths.txt`
    + `test_images_filepaths.txt`
    + `southaf_images_filepaths.txt`
    + `dams.names`
        + contains class names
    + `[run_date].data`
        + contains directory information for darknet
        
#### Modify `yolov3.cfg`

+ Create of a copy `/darknet/cfg/yolov3.cfg` in your directory
+ Modify `yolov3.cfg` file to work on one class, as per detailed at  https://github.com/WyattAutomation/Train-YOLOv3-with-OpenImagesV4#cfg-file:
    + Line 3, set `batch=16`
    + Line 4, set `subdivisions=16`
    + Line 603, set `filters=18`, filters = (number_of_classes + 5) * 3 
    + Line 610, set `classes=1`, we have one class (dams)
    + Line 689, set `filters=18`, filters = (number_of_classes + 5) * 3 
    + Line 696, set `classes=1`, we have one class (dams)
    + Line 776, set `filters=18`, filters = (number_of_classes + 5) * 3 
    + Line 783, set `classes=1`, we have one class (dams)
    + we set 18 filters because each cell in YOLOv3 predicts 3 bounding boxes. Each bounding box has 5 + number_of_classes attributes (dimenions, objectness score, class confidence)
+ You can also modify `max_batches=` on line 20 to limit the number of iterations. For only 1 class, 2000 should be the minumum for this, but many have put at least 4000 to be safe
    + This is because the learning rate usually decreases to 0.0001 at around 2000 * num_class iterations. 
+ Modify name of cfg file to `[run_date].cfg` to keep track of edits
        
## Training Procedure

#### Required files:
+ Classes (`dams.names`):
   + contains class names, there should be only one, dam
+ Directory Paths (`[run_date].data`):
   + contains directory paths for darknet
   + this includes paths for `train.txt`, `validation.txt`, and `backup`
   + IMPORTANT: modify name of `[run_date].data` for each run, to keep track of past runs
        + for example, `06-24.data` or `06-25.data`
+ Modified `[run_date].cfg` (see above)
    + IMPORTANT: modify name of `[run_date].cfg` for each run, because the file names for the output weights are named after the input `.cfg` file
        + for example, `06-24.cfg` or `06-25.cfg`
+ Image directory
   + `/images`, paths to each individual image should be in `train.txt` and `validation.txt`
+ Labels directory
   + `/labels`, paths to each label should correspond to an image path
+ pre- trained weights: `darknet53.conv.74` (pre-trained on ImageNet)
   + The weights will be different if you are resuming training you paused earlier. In this case, use the most recent OUTPUT of the training (the checkpoint): `dams_last.weights`
   
#### Required directories:
+ `/data` should contain the images, labels, `.txt` files, and `dams.names`
+ `/cfg` should contain `[run_date].data` and `[run_date].cfg`

#### Train model

Important Information:
+ `darknet` is an application written in C and CUDA
+ To train a model, use `./darknet detector train`
    + To detect objects in an images, use `./darknet detect` instead
+ `darknet`recieves the architecture in the input `.cfg` file. In this case, `dams.cfg` provides a modified `yolov3.cfg` architecture for one class 
+ To track the loss after each training batch, include ` > /path/to/train.log` at the end of the training command above
    + Use `grep "avg" /path/to/training_loss.log` to monitor the average loss and learning rate
    + Once the learning rate reaches a small number (~0.0001), you could stop training
    
Run `./darknet detector train /path/to/dams.data /path/to/dams.cfg /path/to/weights`
+ Example: `./darknet detector train cfg/[run_date].data cfg/[run_date].cfg /darknet/darknet53.conv.74`
+ To monitor loss: `./darknet detector train cfg/[run_date].data cfg/[run_date].cfg /darknet/darknet53.conv.74 > /path/to/outputs/training_loss.log`

#### Outputs
+ Every 100 iterations, `dams_last.weights` checkpoint will be saved to the `backup` directory listed in `dams.data`
    + It is possible to stop training and then resume using `dams_last.weights` as the input weights
+ Every 10000 iterations, `dams_xxxxx.weights` will be saved to `backup` 
+ When training is complete, `dams_final.weights` will be saved to `backup`

#### Evaluation Metrics
+ Darknet currently does not compute mAP
    + However, this fork of darknet does: https://github.com/AlexeyAB/darknet
