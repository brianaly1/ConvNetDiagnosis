import pickle
import visualize
import numpy as np
import tensorflow as tf
import os

TFRDIR = '/home/alyb/data/tfrecords/'
PICKLEDIR = '/home/alyb/data/pickles/false_pos.p'
FILE_LIST = os.listdir(TFRDIR)

def loadPickle(file_name):
    with open(file_name,'rb') as openfile:
        while True:
            try:
                example = pickle.load(openfile)
                visualize.visSlice(example[1][0])
            except EOFError:
                print("end of file")
                sys.exit()

def loadTFR(file_name):
    volumes = []
    labels = []
    reader = tf.TFRecordReader
    record_iterator = tf.python_io.tf_record_iterator(file_name)
    for serialized_example in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
     
        volume_raw = example.features.feature["volume_raw"].bytes_list.value[0]
        label = example.features.feature["label"].int64_list.value[0]
      
        volume_1d = np.fromstring(volume_raw)
        volume = volume_1d.reshape((32,32,32))
        volumes.append(volume)
        labels.append(label)
        print(label)
        visualize.visSlice(volume[16])
    
    volumes = np.array(volumes)
    labels= np.array(labels)
    print(np.shape(volumes))

def _read_bulk(file_names):
    '''
    Read tfrecord files and combine all examples

    Inputs:
        file_names: list of tf record files
    Outputs:
        volumes: a tensor of size [num examples in files, 32, 64 , 64 ,1]
        labels: a tensor of size [num examples in files]
    '''
    volumes = []
    labels = []
    reader = tf.TFRecordReader
    for file_name in file_names:
        record_iterator = tf.python_io.tf_record_iterator(file_name)
        for serialized_example in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            volume_raw = example.features.feature["volume_raw"].bytes_list.value[0]
            label = example.features.feature["label"].int64_list.value[0]
            # create volume array
            volume_1d = np.fromstring(volume_raw)
            volume = volume_1d.reshape((32,64,64))
            volumes.append(volume)
            labels.append(label)
    labels = np.array(labels)
    labels = np.expand_dims(label,-1)
    volumes = np.array(volumes)
    volumes = np.expand_dims(volumes,-1)

    return volumes,labels

file_names = os.path.join(TFRDIR,FILE_LIST[0])

loadPickle(PICKLEDIR)
