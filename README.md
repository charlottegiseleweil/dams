This repo contains the work in progress Deep Learning part of [NatCap](https://naturalcapitalproject.stanford.edu/) *Detecting Dams with AI and Satellite Imagery*. 
It is funded by the National Geogrpahic and Microsoft AI for Earth program. 

# Imagery
See https://bitbucket.org/natcap/extract-dams/src/default/

# Training (TFOD)

See readme in [dams/tfod/training/](https://github.com/charlottegiseleweil/dams/tree/master/tfod/training)
```python model_train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/{{CONFIG FILE}}.config
```

# Inference

1) Export frozen graph with [dams/tfod/detection/export_inference_graph.py](https://github.com/charlottegiseleweil/dams/tree/master/tfod/detection)
```python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ../../../../repos/dams/tfod/training/configs/07_29_imagery7-25_Faster_rcnn_resnet50_coco.config \
    --trained_checkpoint_prefix ../../../../outputs/fasterRCNN_07_27_newimagery/model.ckpt-300000 \
    --output_directory ../../../../outputs/fasterRCNN_07_27_newimagery/export_inference \
    --write_inference_graph True
```

2) Run detector [dams/tfod/detection/run_detector_to_bboxes.py](https://github.com/charlottegiseleweil/dams/tree/master/tfod/detection/run_detector_to_bboxes.py)

Playground/Visualize inference on a few images: [Inference notebook](https://github.com/charlottegiseleweil/dams/blob/master/tfod/detection/Inference.ipynb)

# Training & Inference (Darknet)
See [dams/yolov3/readme_ultralytics.md](https://github.com/charlottegiseleweil/dams/blob/master/yolov3/readme_ultralytics.md)

# Evaluation
[Work in progress]
