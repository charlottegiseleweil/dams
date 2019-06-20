### Code to fix issue "DataLossError (see above for traceback): truncated record at"
###	
### Charlotte Gisèle Weil May 30 2019
### from https://www.kaggle.com/c/youtube8m/discussion/30208

import tensorflow as tf
import glob
import os

files =  glob.glob("~/charlie/data/05-29_cleaned/*.tfrecord") 

filesSize = len(files)
cnt = 0 

for filename in files:
    cnt = cnt + 1
    print('checking %d/%d %s' % (cnt, filesSize, filename))
    try:
        for example in tf.python_io.tf_record_iterator(filename): 

            tf_example = tf.train.Example.FromString(example) 

    except:
        print("removing %s" % filename)
        os.remove(filename)