#This is a sample Image 
FROM tensorflow/tensorflow:1.9.0-gpu-py3 

RUN mkdir /lib/tf
COPY set_up_object_detection_image.sh /lib/tf/set_up_object_detection_image.sh
RUN sh /lib/tf/set_up_object_detection_image.sh

CMD [“echo”,”Image created”] 