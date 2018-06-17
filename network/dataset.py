import tensorflow as tf
import numpy as np
import os
import pickle
import sys
sys.path.insert(0,'/home/alyb/ConvNetDiagnosis/processing')
import extract
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

def create_dataset_test(volumes_path,save_dir,sample_spacings):
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
            all_sub_vols = []
            pixels = volume[0]
            centroids = volume[1]
            spacing = volume[2]
            vol_shape = np.shape(pixels)
            sub_vol_shape = np.array([32,32,32])
            sub_vol_centroids,labels = extract.segmentVolume(vol_shape,centroids,sub_vol_shape)
            for new_spacing in sample_spacings: 
                sub_vols,patches = extract.extractCandidate(volume,sub_vol_centroids,sub_vol_shape,spacing,new_spacing,translate=0)
                all_sub_vols.append(sub_vols)
        print(len(all_sub_vols)) 
        sys.exit()
                    
def create_dataset(files,desired_counts,labels,save_dir):
    '''
    Load data from the serialized files and chunk into mini batches
    mini batches will be passed to creatDataset() to be saved as TFRecord files
    which will be loaded as tensorflow dataset objects when training
    
    Inputs:
        files: list of paths to files, all elements in a file have the same label
        desired_counts: number of examples to be included from each file
        labels: list of label for examples contained in each file
        save_dir: path to save directory
    '''
    file_pointers = [0,0,0]
    name_count = 0
    while True:
        mini_batch = []
        labels = []
        try:
            for index,file_name in enumerate(files):
                with open(file_name,'rb') as openfile:
                    sub_vols = []
                    count = 0
                    openfile.seek(file_pointers[index])
                    while count < desired_counts[index]:
                        subvol_list = pickle.load(openfile)[0]
                        sub_vols.extend(subvol_list)
                        labels.extend([labels[index]]*len(subvol_list))
                        count = count + len(subvol_list)
                    file_pointers[index] = openfile.tell()
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


def main():
    sample_spacings = [[1.5,0.5,0.5],[1.375,0.625,0.625],[1.25,0.75,0.75]]
    create_dataset_test('/home/alyb/data/pickles/volumes.p','/home/alyb/data/tfrecords/tfrecords-test/',sample_spacings)

if __name__=="__main__":
    main()

