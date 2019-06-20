## Install

darknet installation:

```
git clone https://github.com/pjreddie/darknet.git
cd darknet
make
```
download the weights

```
wget https://pjreddie.com/media/files/yolov3.weights
```

## Pre-processing data

#### prepare_darknet_training_data.ipynb
inputs: dams_images.zip and not_a_dam_images.zip
outputs: images folder + labels folder, each with test and tr
labels formats: image id.txt
    ….

+ create name file
    dams.names:
        dam

+ create data file
    dams.data
        …

## Running the model
+ dams.cfg
    modified version of ???
+ dams.data

### Pre-trained model: ImageNet.
