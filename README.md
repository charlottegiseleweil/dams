This repo contains the work in progress Deep Learning part of [NatCap](https://naturalcapitalproject.stanford.edu/) *Detecting Dams with AI and Satellite Imagery*. 
It is funded by the National Geogrpahic and Microsoft AI for Earth program. 

# Imagery
See https://bitbucket.org/natcap/extract-dams/src/default/

# Training (TFOD)

See readme in [dams/tfod/training/](https://github.com/charlottegiseleweil/dams/tree/master/tfod/training)
```
python model_train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/{{CONFIG FILE}}.config
```

# Inference (TFOD)

1) Export frozen graph with [dams/tfod/detection/export_inference_graph.py](https://github.com/charlottegiseleweil/dams/tree/master/tfod/detection)
```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ../../../../repos/dams/tfod/training/configs/07_29_imagery7-25_Faster_rcnn_resnet50_coco.config \
    --trained_checkpoint_prefix ../../../../outputs/fasterRCNN_07_27_newimagery/model.ckpt-300000 \
    --output_directory ../../../../outputs/fasterRCNN_07_27_newimagery/export_inference \
    --write_inference_graph True
```

2) Run detector [inference.py](https://github.com/charlottegiseleweil/dams/tree/master/tfod/detection/inference.py)

Playground/Visualize inference on a few images: [Inference notebook](https://github.com/charlottegiseleweil/dams/blob/master/tfod/detection/Inference.ipynb)

# Training & Inference (Darknet)
See [dams/yolov3/readme_ultralytics.md](https://github.com/charlottegiseleweil/dams/blob/master/yolov3/readme_ultralytics.md)

# Evaluation
1) Store results table (a row per observation), with column Results@IoU@Conf with TP, FP, TN, FN at specific IoU and minimum confidence threshold.
* results_{set}_{model}.csv with for one row per image
* results_sparse for one row per observation (duplicate images that have FP+FN) [make_results_table]

2) Calculate custom Evaluation metric: Recall @ 5% IoU and min confidence threshold = ??? [evaluation.py]

3) Visualize images