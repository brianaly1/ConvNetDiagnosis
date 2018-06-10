import sys
import os
import numpy as np
import tensorflow as tf

NUM_GPUS = 1
GPUS = ['/gpu:1']
BATCH_SIZE = 100
CHECKPOINT = 

def partition(volumes):

def predict(sub_vols): #load graph...
    sub_vols = np.stack(sub_vols)
    sub_vols = tf.convert_to_tensor(sub_vols,dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(sub_vols)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator() 
    with tf.variable_scope(tf.get_variable_scope()):  
        with tf.device(GPUS[0]):
            with tf.name_scope('%s_%d_%s' % ('tower', 0,'infer')) as scope: 
                volume_batch = iterator.get_next()
                logits = c3d.inference(volume_batch,BATCH_SIZE)    
                predictions = tf.sigmoid(logits)
                predictions = tf.round(predictions)
                predictions = tf.cast(predictions,dtype=tf.int32)

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
    sess.run(iterator.initializer)
    saver.restore(sess, CHECKPOINT)   

    all_predictions = []
    while True:
        try:
            pred_vals = sess.run(predictions)
            all_predictions.extend(pred_vals)
        except tf.errors.OutOfRangeError:
            print("inference complete")
            break

    return(all_predictions)
 
def main():
    scans = [] #filenames
    all_nodules = []
    for scan in scans:
        nodules = []
        volume = ()
        sub_vols = partition(volume)
        predictions = predict(sub_vols)
        for i,prediction in enumerate(predictions):
            if prediction == 1:
                nodules.append(sub_vols[i])
        all_nodules.append(nodules)

    #for scan in all_nodules:
         #for nodule in scan:
            #visualize.visVol(nodule)
   
