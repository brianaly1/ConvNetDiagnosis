import sys
import os
import numpy as np
import tensorflow as tf

NUM_GPUS = 1
GPUS = ["/gpu:1"]
BATCH_SIZE = 100
CHECKPOINT = "/home/alyb/trained_models/checkpoints/96val/" 
DATA_DIR = "/home/alyb/data/tfrecords/"

def _parse_function(example_proto):
    '''
    parse tf record example and convert from strings to arrays
    
    Inputs:
        example_proto:

    '''
    features = {"volume_raw":tf.FixedLenFeature((), tf.string, default_value=''),
                "label":tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, features)
    volume_raw = parsed_features["volume_raw"]
    label = tf.reshape(parsed_features["label"],tf.stack([1]))
    label = tf.cast(label,tf.float32)
    volume_1d = tf.decode_raw(volume_raw,tf.float64)
    volume = tf.reshape(volume_1d,tf.stack(VOL_SHAPE))
    volume = tf.expand_dims(volume,-1)
    volume = tf.cast(volume,tf.float32)
    return volume,label

def predict(sub_vols,files): #load graph...
    labels = None
    if sub_vols not None:
        sub_vols = np.stack(sub_vols)
        sub_vols = tf.convert_to_tensor(sub_vols,dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices(sub_vols)
    else:
        dataset = tf.data.TFRecordDataset(files)
        dataset_val.map(_parse_function)

    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_initializable_iterator() 
    with tf.variable_scope(tf.get_variable_scope()):  
        with tf.device(GPUS[0]):
            with tf.name_scope('infer') as scope: 
                if sub_vols not None:
                    volume_batch = iterator.get_next()
                else:
                    volume_batch,labels = iterator.get_next()
                logits = c3d.inference(volume_batch,BATCH_SIZE,tf.constant(False,dtype=tf.bool))    
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

    return(all_predictions,labels)

def test():
    data_files = os.listdir(DATA_DIR)  
    test_set = data_files[370:390]
    test_paths = list(map(lambda file: os.path.join(DATA_DIR,file),test_set))
    predictions,labels = predict(None,test_paths)
    print(predictions)
    
#def main():
#    scans = [] #filenames
#    all_nodules = []
#    for scan in scans:
#        nodules = []
#        volume = ()
#        sub_vols = partition(volume)
#        predictions = predict(sub_vols)
#        for i,prediction in enumerate(predictions):
#            if prediction == 1:
#                nodules.append(sub_vols[i])
#        all_nodules.append(nodules)

    #for scan in all_nodules:
         #for nodule in scan:
            #visualize.visVol(nodule)

    
