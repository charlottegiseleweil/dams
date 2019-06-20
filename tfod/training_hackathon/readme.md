### Directory orga: 

* annotations: This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images.

* images: This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.

* images\train: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model.
images\test: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model.

* pre-trained-model: This folder will contain the pre-trained model of our choice, which shall be used as a starting checkpoint for our training job.

* training: This folder will contain the training pipeline configuration file *.config, as well as a *.pbtxt label map file and all files generated during the training of our model.


### Run
python model_train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/{{CONFIG FILE}}.config

### References
[TF Object Detection API Tuto](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#annotating-images)
