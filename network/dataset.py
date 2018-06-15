import tensorflow as tf
import numpy as np
import os
import pickle
import sys

DATA_DIR = '/home/alyb/data/luna16/'
VOLUMES_FILE = '/home/alyb/data/pickles/volumes.p'

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _shuffle(X,Y):
  p = np.random.permutation(len(Y))
  return(X[p],Y[p])

def saveTf(volumes,labels,file_name):
    '''
    Save TFRecord files with data and labels
    Inputs:
        volumes: np array of voxels size [num_examples, height, rows, columns]
        labels:  np array of binary labels of size [num_examples]
        file_name: .tfrecords file path to save to
    '''
    num_examples = np.shape(volumes)[0]
    assert num_examples == np.shape(labels)[0] , "volume array size does not match labels array size"

    print('Writing', file_name)

    with tf.python_io.TFRecordWriter(file_name) as writer:
        for i in range(num_examples):
            volume_raw = volumes[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                                       'label': _int64_feature(int(labels[i])),
                                       'volume_raw': _bytes_feature(volume_raw)
                                                                             }))
            writer.write(example.SerializeToString())

def create_dataset_test(volumes_path,save_dir):
    '''
    Segment volumes into sub volumes
    save subvols for each vol in a seperate tfrecords file
    Inputs:
        volumes_path: path to pickle file containing volumes - [pixels,centroid_list]
        save_dir: path to directory to save tfrecord files in
    '''
    with open(volumes_path,"rb") as openfile:
        while True:
            volume = pickle.load(openfile)
            pixels = volume[0]
            centroids = volume[1]
            subvolumes,labels = segment_vol(pixels,centroids)
            sys.exit()
                    
       











def create_dataset(pos_path,falsepos_path,random_path,save_dir):
    '''
    Load data from the serialized files and chunk into mini batches
    mini batches will be passed to creatDataset() to be saved as TFRecord files
    which will be loaded as tensorflow dataset objects when training
    
    Inputs:
        pos_path: path to file with serialized positive sub volumes 
        falsepos_path: path to file with serialized false positives
        random_path: path to file with randomly extracted sub volumes
        save_dir: path to save directory
    '''
    file1_pointer = 0
    file2_pointer = 0
    file3_pointer = 0
    name_count = 0
    while True:
        mini_batch = []
        labels = []
        try:
            with open(pos_path,'rb') as openfile:
                sub_vols = []
                count = 0
                openfile.seek(file1_pointer)
                while count < 180:
                    subvol_list = pickle.load(openfile)[0]
                    sub_vols.extend(subvol_list)
                    labels.extend([1]*len(subvol_list))
                    count = count + len(subvol_list)
                file1_pointer = openfile.tell()
                mini_batch.extend(sub_vols) 
            with open(falsepos_path,'rb') as openfile:
                sub_vols = []
                count = 0
                openfile.seek(file2_pointer)
                while count < 800:
                    subvol_list = pickle.load(openfile)[0]
                    sub_vols.extend(subvol_list)
                    labels.extend([0]*len(subvol_list))
                    count = count + len(subvol_list)
                file2_pointer = openfile.tell()
                mini_batch.extend(sub_vols)
            with open(random_path,'rb') as openfile:
                sub_vols = []
                count = 0
                openfile.seek(file3_pointer)
                while count < 20:
                    subvol_list = pickle.load(openfile)[0]
                    sub_vols.extend(subvol_list)
                    labels.extend([0]*len(subvol_list))
                    count = count + len(subvol_list)
                file3_pointer = openfile.tell()
                mini_batch.extend(sub_vols)
            mini_batch = np.array(mini_batch)
            labels = np.array(labels)
            mini_batch,labels = _shuffle(mini_batch,labels)

            name_count = name_count + 1
            file_name = 'batch_' + str(name_count) + '.tfrecords'
            save_path = os.path.join(save_dir,file_name)
            saveTf(mini_batch,labels,save_path)

            print('current data count is: %d' %(len(mini_batch)*name_count))
            print('current label count is: %d' %(len(labels)*name_count))
        except EOFError:
            print('done')
            sys.exit()
        except KeyboardInterrupt:
            print('Interrupted')
            sys.exit()
        except IOError:
            print("I/O exception, most likely ran out of disk space")
            sys.exit()
        except:
            print('unknown error') 
            sys.exit()       
        print('--------')




