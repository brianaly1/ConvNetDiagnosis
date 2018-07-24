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

def saveTf(sub_volumes,centroids,labels,file_name,mode,mag=1):
    '''
    Save TFRecord files with data and labels
    Inputs:
        sub_volumes: np array of voxels size [num_examples, height, rows, columns]
        centroids: np array of centers of each sub volume wrt to the overall patient volume
        labels:  np array of binary labels of size [num_examples]
        file_name: index used to give each file a unique name
        mode: 0 for LUNA, 1 for NDSB
    '''

    save_path = os.path.join(settings.CA_TRAIN_DATA_DIR,"TFRecordsTest")

    if mode==0:
        save_path = os.path.join(save_path,"LUNA",file_name+".tfrecords")
    elif mode==1:
        val1_files = set(os.listdir(os.path.join(save_path,"NDSBVAL","1")))
        if (file_name+".tfrecords") in val1_files:
            save_path = os.path.join(save_path,"NDSBVAL",str(mag),file_name+".tfrecords")
        else:
            save_path = os.path.join(save_path,"NDSB",str(mag),file_name+".tfrecords")

    num_examples = np.shape(sub_volumes)[0]
    assert num_examples == np.shape(labels)[0] , "volume array size does not match labels array size"

    print('Writing', save_path)

    with tf.python_io.TFRecordWriter(save_path) as writer:
        for i in range(num_examples):
            volume_raw = sub_volumes[i].tostring()
            centroid_raw = centroids[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                                       'label': _int64_feature([int(labels[i])]),
                                       'volume_raw': _bytes_feature(volume_raw),
                                       'centroid_raw': _bytes_feature(centroid_raw)
                                                                             }))
            writer.write(example.SerializeToString())

def save_luna_test_data(patient,sub_vol_shape):
    '''
    Generate TFRecord files, one per patient, containing all relevant sub volumes in the volume
    Inputs:
        patient: patient uid
        sub_vol_shape: desired sub volume shape
    '''

    nodules = utils.get_patient_nodules(patient)
   
    sub_vols,centroids,labels = utils.partition_volume(patient, sub_vol_shape, mag=1, centroids=nodules)

    sub_vols_np = np.array(sub_vols)
    centroids_np = np.array(centroids)
    labels_np = np.array(labels)

    pos_indices = np.nonzero(labels_np)
    pos_centroids = centroids_np[pos_indices]

    print("processing patient: {}, with {} subvolumes, {} nodules, and {} positive labels".format(patient,len(sub_vols),len(nodules),len(pos_indices[0])))
    saveTf(sub_vols_np,centroids_np,labels_np,patient,mode=0)
    return len(nodules),len(pos_indices[0]) 

def save_ndsb_test_data(patient,sub_vol_shape,label,mag):
    '''
    Generate TFRecord files, one per patient, containing all relevant sub volumes in the volume
    Inputs:
        patient: patient uid
        sub_vol_shape: desired sub volume shape
        label: label for the given patient (cancer/no)
    '''

    sub_vols,centroids,_ = utils.partition_volume(patient, sub_vol_shape, mag=mag, centroids=None)
    labels = [label]*len(sub_vols)
    sub_vols_np = np.array(sub_vols)
    centroids_np = np.array(centroids)
    labels_np = np.array(labels)

    print("processing patient: {}, with {} subvolumes".format(patient,len(sub_vols)))
    saveTf(sub_vols_np,centroids_np,labels_np,patient,mode=1,mag=mag)
    

def luna_main():

    labels_dir = os.path.join(settings.CA_PATIENTS_DIR,"labels")
    patients = os.listdir(labels_dir)
    sub_vol_shape = settings.SUB_VOL_SHAPE
    tot_nods = 0
    tot_pos = 0
    for patient in patients_list:
        try:
            nods,pos = save_luna_test_data(patient,sub_vol_shape)
            tot_nods+=nods
            tot_pos+=pos
            print("Total nodules = {}, total positive labels = {}".format(tot_nods,tot_pos))
            print('--------------------------------------------------------------------------')
        except KeyboardInterrupt:
            print("Interrupted")
            sys.exit()
        except:
            print("Unknown error with patient {}".format(patient))
    
def ndsb_main(mag=1):

    patients = utils.load_ndsb_patients()
    sub_vol_shape = settings.SUB_VOL_SHAPE
    for patient in patients:
        try:
            save_ndsb_test_data(patient,sub_vol_shape,patients[patient],mag=mag)
        except KeyboardInterrupt:
            print("Interrupted")
            sys.exit()
        except Exception as e:
            print("Error with patient {} : {}".format(patient,e))
    
if __name__=="__main__":
    ndsb_main(mag=2)
    

     
         
