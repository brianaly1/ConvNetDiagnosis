from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import numpy as np
import tensorflow as tf
import c3d
sys.path.insert(0, '/home/alyb/ConvNetDiagnosis/processing/')
import settings
import utils
import visualize

NUM_GPUS = 1
GPUS = ["/gpu:7"]
BATCH_SIZE = 100 
VOL_SHAPE = [32,32,32]

def _parse_function(example_proto):
    '''
    parse tf record example and convert from strings to arrays
    
    Inputs:
        example_proto:

    '''
    features = {"volume_raw":tf.FixedLenFeature((), tf.string, default_value=''),
                "centroid_raw":tf.FixedLenFeature((), tf.string, default_value=''),
                "label":tf.FixedLenFeature((), tf.int64, default_value=0)}


    parsed_features = tf.parse_single_example(example_proto, features)
    volume_raw = parsed_features["volume_raw"]
    center_raw = parsed_features["centroid_raw"]
    label = parsed_features["label"]
    volume_1d = tf.decode_raw(volume_raw,tf.float64)
    center_1d = tf.decode_raw(center_raw,tf.int64)
    volume = tf.reshape(volume_1d,tf.stack(VOL_SHAPE))
    volume = tf.expand_dims(volume,-1)
    volume = tf.cast(volume,tf.float32)
    volume = volume / 255
    
    return volume,label

def predict(files): #load graph...

    CHECKPOINT = os.path.join(settings.TRAIN_DATA_DIR,"Checkpoints")
    
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        filenames = tf.placeholder(tf.string, shape=[None])

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_initializable_iterator() 

        with tf.variable_scope(tf.get_variable_scope()):  
            with tf.device(GPUS[0]):
                with tf.name_scope('infer') as scope: 
                    volume_batch,labels = iterator.get_next()
                    logits = c3d.inference(volume_batch,BATCH_SIZE,tf.constant(False,dtype=tf.bool))      
                    predictions = tf.sigmoid(logits)
                    #predictions = tf.round(predictions)
                    #predictions = tf.cast(predictions,dtype=tf.int32)

        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        sess.run(iterator.initializer,feed_dict={filenames:files})
        saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT))   

        all_predictions = []
        all_labels = []
        all_volumes = []

        while True:
            try:
                pred_vals,label_vals,vol_vals = sess.run([predictions,labels,volume_batch])
                all_predictions.extend(pred_vals)
                all_labels.extend(label_vals)
                all_volumes.extend(volume_batch)
            except tf.errors.OutOfRangeError:
                print("inference complete")
                all_predictions = np.array(all_predictions,dtype=np.float32)
                all_predictions = np.squeeze(all_predictions)
                all_labels = np.array(all_labels)
                all_labels = np.squeeze(all_labels)  
                all_volumes = np.array(all_volumes)      
                break

    return(all_predictions,all_labels,all_volumes)

def mine_fps(test_paths):

    for path in test_paths:

        counter = 0

        predictions_np,labels_np,volumes_np = predict([path])
        predictions_np[predictions_np > settings.THRESHOLD] = 1
        predictions_np[predictions_np <= settings.THRESHOLD] = 0
        predictions_np = predictions_np.astype(np.int32)

        uid = path.split('/')[-1][:-10]
        dst_dir = os.path.join(settings.TRAIN_DATA_DIR,"FP")
        target_path = os.path.join(dst_dir,uid + "_" + str(counter) + "_0_" + "fp.png")

        pred_list = predictions_np.tolist()
        label_list = labels_np.tolist()  
        vol_list = volumes_np.tolist()
      
        for index in range(len(label_list)):
            if pred_list[index] == 1 and label_list[index]==0:
                utils.save_cube_img(target_path, vol_list[index], 4, 8)
                sys.exit()
                counter+=1

        print(counter)

    
def main():
    data_dir = os.path.join(settings.TRAIN_DATA_DIR,"TFRecordsTest")
    tf_data_files = os.listdir(data_dir)  
    tf_test_set = tf_data_files[0:1]
    tf_test_paths = list(map(lambda file: os.path.join(data_dir,file),tf_test_set))
    mine_fps(tf_test_paths)

if __name__=="__main__":
    main()    
