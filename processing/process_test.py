import sys
import os
import utils
import settings
import numpy as np
import tensorflow as tf

def _int64_feature(value):
  val_list = []
  val_list.extend(value)
  return tf.train.Feature(int64_list=tf.train.Int64List(value=val_list))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def saveTf(sub_volumes,centroids,labels,file_name):
    '''
    Save TFRecord files with data and labels
    Inputs:
        sub_volumes: np array of voxels size [num_examples, height, rows, columns]
        centroids: np array of centers of each sub volume wrt to the overall patient volume
        labels:  np array of binary labels of size [num_examples]
        file_index: index used to give each file a unique name
    '''
    save_path = os.path.join(settings.TRAIN_DATA_DIR,"TFRecordsTest",file_name+".tfrecords")
    num_examples = np.shape(sub_volumes)[0]
    assert num_examples == np.shape(labels)[0] , "volume array size does not match labels array size"

    print('Writing', save_path)

    with tf.python_io.TFRecordWriter(save_path) as writer:
        for i in range(num_examples):
            volume_raw = sub_volumes[i].tostring()
            centroid = [int(x) for x in centroids[i]]
            example = tf.train.Example(features=tf.train.Features(feature={
                                       'label': _int64_feature([int(labels[i])]),
                                       'volume_raw': _bytes_feature(volume_raw),
                                       'center': _int64_feature(centroid)
                                                                             }))
            writer.write(example.SerializeToString())

def save_patient_test_data(patient,sub_vol_shape):
    '''
    Generate TFRecord files, one per patient, containing all relevant sub volumes in the volume
    Inputs:
        patient: patient uid
    '''

    nodules = utils.get_patient_nodules(patient)
   
    sub_vols,centroids,labels = utils.partition_volume(patient, nodules, sub_vol_shape, mag=1)

    sub_vols_np = np.array(sub_vols)
    centroids_np = np.array(centroids)
    labels_np = np.array(labels)

    pos_indices = np.nonzero(labels_np)
    pos_centroids = centroids_np[pos_indices]

    print("processing patient: {}, with {} subvolumes, {} nodules, and {} positive labels".format(patient,len(sub_vols),len(nodules),len(pos_indices[0])))
    saveTf(sub_vols_np,centroids_np,labels_np,patient)
    return len(nodules),len(pos_indices[0]) 

def main():

    labels_dir = os.path.join(settings.PATIENTS_DIR,"labels")
    patients_list = os.listdir(labels_dir)
    sub_vol_shape = settings.SUB_VOL_SHAPE
    tot_nods = 0
    tot_pos = 0
    for patient in patients_list:
        try:
            nods,pos = save_patient_test_data(patient,sub_vol_shape)
            tot_nods+=nods
            tot_pos+=pos
            print("Total nodules = {}, total positive labels = {}".format(tot_nods,tot_pos))
            print('--------------------------------------------------------------------------')
        except KeyboardInterrupt:
            print("Interrupted")
            sys.exit()
        except:
            print("Unknown error with patient {}".format(patient))
    

if __name__=="__main__":
    main()
    

     
         
