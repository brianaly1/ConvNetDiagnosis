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
    
    return volume,label,center_1d

def predict(files,mode): #load graph...

    CHECKPOINT = os.path.join(settings.TRAIN_DATA_DIR,"Checkpoints")
    if mode==0:
        CHECKPOINT = os.path.join(CHECKPOINT,"roi")
    elif mode==1:
        CHECKPOINT = os.path.join(CHECKPOINT,"mal")    
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        filenames = tf.placeholder(tf.string, shape=[None])

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_initializable_iterator() 

        with tf.variable_scope(tf.get_variable_scope()):  
            with tf.device(GPUS[0]):
                with tf.name_scope('infer') as scope: 
                    volume_batch,labels,centers = iterator.get_next()
                    logits = c3d.inference(volume_batch,BATCH_SIZE,tf.constant(False,dtype=tf.bool),mode)      
                    predictions = tf.sigmoid(logits)
                    #predictions = tf.round(predictions)
                    #predictions = tf.cast(predictions,dtype=tf.int32)

        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))
        sess.run(iterator.initializer,feed_dict={filenames:files})
        saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT))   

        all_predictions = []
        all_labels = []
        all_volumes = []
        all_cens = []

        while True:
            try:
                pred_vals,label_vals,vol_vals,cen_vals = sess.run([predictions,labels,volume_batch,centers])
                all_predictions.extend(pred_vals)
                all_labels.extend(label_vals)
                all_volumes.extend(vol_vals)
                all_cens.extend(cen_vals)
            except tf.errors.OutOfRangeError:
                print("inference complete")
                print("--------------------")
                all_predictions = np.array(all_predictions,dtype=np.float32)
                all_predictions = np.squeeze(all_predictions)
                all_labels = np.array(all_labels)
                all_labels = np.squeeze(all_labels)  
                all_cens = np.array(all_cens)
                all_cens = np.squeeze(all_cens) 
                all_volumes = np.array(all_volumes)      
                break

    return(all_predictions,all_labels,all_volumes,all_cens)

def test(test_paths,mode=0,mine_fps=0):

    fps = [] 
    fns = []
    tps = []
    tns = []

    file_counter = 0

    for path in test_paths:

        fp_counter = 0

        fps.append([])
        fns.append([])
        tps.append([])
        tns.append([])

        predictions_np,labels_np,volumes_np,cens_np = predict([path],mode=mode)
        predictions_np[predictions_np > settings.THRESHOLD] = 1
        predictions_np[predictions_np <= settings.THRESHOLD] = 0
        predictions_np = predictions_np.astype(np.int32)
        acc = np.mean((predictions_np == labels_np).astype(np.float32)) 
        uid = path.split('/')[-1][:-10]
        dst_dir = os.path.join(settings.TRAIN_DATA_DIR,"FP")
        for index in range(np.shape(labels_np)[0]):
            target_path = os.path.join(dst_dir,uid + "_" + str(fp_counter) + "_0_" + "fp.png")
            if predictions_np[index] == 1 and labels_np[index]==0:
                fps[-1].extend(cens_np[index])
                if mine_fps:
                    volume = np.squeeze(volumes_np[index])
                    utils.save_cube_img(target_path, volume*255, 4, 8)
                    fp_counter+=1
            elif predictions_np[index] == 0 and labels_np[index]==1:
                fns[-1].extend(cens_np[index])
            elif predictions_np[index] == 1 and labels_np[index]==1:
                tps[-1].extend(cens_np[index])
            elif predictions_np[index] == 0 and labels_np[index]==0:
                tns[-1].extend(cens_np[index])
        
        file_counter += 1
        print("{} --- acc is: {}, fps: {}, fns: {},tps: {},tns: {}".format(file_counter,acc,len(fps[-1]),len(fns[-1]),len(tps[-1]),len(tns[-1])))

def main():
    data_dir = os.path.join(settings.TRAIN_DATA_DIR,"TFRecordsTest")
    tf_data_files = os.listdir(data_dir)  
    tf_test_set = tf_data_files[0:8]
    tf_test_paths = list(map(lambda file: os.path.join(data_dir,file),tf_test_set))
    test(tf_test_paths)

if __name__=="__main__":
    main()    
