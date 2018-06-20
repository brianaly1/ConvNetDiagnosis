import sys
import os
import numpy as np
import tensorflow as tf
import c3d

NUM_GPUS = 1
GPUS = ["/gpu:1"]
BATCH_SIZE = 100
CHECKPOINT = '/home/alyb/ConvNetDiagnosis/network/checkpoints/'
TF_DATA_DIR = "/home/alyb/Data/tfrecords/tfrecords-test/"
VOL_SHAPE = [32,32,32]

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

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        filenames = tf.placeholder(tf.string, shape=[None])

        if sub_vols != None:
            sub_vols = np.stack(sub_vols)
            sub_vols = tf.convert_to_tensor(sub_vols,dtype=tf.float32)
            dataset = tf.data.Dataset.from_tensor_slices(sub_vols)
        else:
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(_parse_function)
    
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_initializable_iterator() 
        with tf.variable_scope(tf.get_variable_scope()):  
            with tf.device(GPUS[0]):
                with tf.name_scope('infer') as scope: 
                    if sub_vols != None:
                        volume_batch = iterator.get_next()
                    else:
                        volume_batch,labels = iterator.get_next()
                    logits = c3d.inference(volume_batch,BATCH_SIZE,tf.constant(False,dtype=tf.bool))    
                    predictions = tf.sigmoid(logits)
                    predictions = tf.round(predictions)
                    predictions = tf.cast(predictions,dtype=tf.int32)

        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        sess.run(iterator.initializer,feed_dict={filenames:files})
        print(CHECKPOINT)
        saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT))   

        all_predictions = []
        all_labels = []
        while True:
            try:
                pred_vals,label_vals = sess.run([predictions,labels])
                all_predictions.extend(pred_vals)
                all_labels.extend(label_vals)
            except tf.errors.OutOfRangeError:
                print("inference complete")
                all_predictions = np.array(all_predictions)
                all_predictions = np.squeeze(all_predictions)
                all_predictions = all_predictions.tolist()
                all_labels = np.array(all_labels)
                all_labels = np.squeeze(all_labels)
                all_labels = all_labels.tolist()                
                break

    return(all_predictions,all_labels)

def test(test_paths):
    predictions,labels = predict(None,test_paths)
    predictions = np.array(predictions,dtype=np.int32)
    labels = np.array(labels,dtype=np.int32)
    element_acc = np.equal(labels,predictions)
    element_acc = element_acc.astype(np.float32)
    mean_acc = np.mean(element_acc)
    print(mean_acc)
    
def main():
    tf_data_files = os.listdir(TF_DATA_DIR)  
    tf_test_set = tf_data_files[2:3]
    tf_test_paths = list(map(lambda file: os.path.join(TF_DATA_DIR,file),tf_test_set))
    test(tf_test_paths)

if __name__=="__main__":
    main()    
