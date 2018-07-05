import os
import sys
import settings
import utils
import tensorflow as tf
import numpy as np

def _int64_feature(value):
  val_list = []
  val_list.extend(value)
  return tf.train.Feature(int64_list=tf.train.Int64List(value=val_list))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _shuffle(X,Y):
  p = np.random.permutation(len(Y))
  return(X[p],Y[p])

def saveTf(volumes,labels,file_index,mode):
    '''
    Save TFRecord files with data and labels
    Inputs:
        volumes: np array of voxels size [num_examples, height, rows, columns]
        labels:  np array of binary labels of size [num_examples]
        file_index: each file needs a unique name, index achieves this
    '''

    folder = "roi"
    if mode == 1:
        folder = "mal"

    save_path = os.path.join(settings.TRAIN_DATA_DIR,"TFRecords",folder,str(file_index)+".tfrecords")
    num_examples = np.shape(volumes)[0]
    assert num_examples == np.shape(labels)[0] , "volume array size does not match labels array size"

    print('Writing', save_path)

    with tf.python_io.TFRecordWriter(save_path) as writer:
        for i in range(num_examples):
            volume_raw = volumes[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                                       'label': _int64_feature([int(labels[i])]),
                                       'volume_raw': _bytes_feature(volume_raw)
                                                                             }))
            writer.write(example.SerializeToString())
        
def create_dataset(categories,translations,total_count,vox_size,mode):
    '''
    Load data from the serialized files and chunk into mini batches
    to be saved as TFRecord files

    Inputs:
        categories: list of data folders to use
        total_count: num of examples in a tf record
    '''
    
    base_path = settings.TRAIN_DATA_DIR

    paths = []
    file_names = []
    count_per_tf = []

    if mode==1:
        count_per_tf = [560,1940]

    for index,category in enumerate(categories):
        cat_path = os.path.join(base_path,category)
        paths.append(cat_path)
        file_names.append(os.listdir(cat_path))
    
    file_counter = 0
    while True:
        try:
            mini_batch = []
            mb_labels = []
            for index,category in enumerate(categories):
                for i in range(0,count_per_tf[index],translations[index]):
                    example = file_names[index].pop()
                    path = os.path.join(paths[index],example)
                    volume = utils.load_cube_img(path, 8, 8, 64)
                    sub_volumes,sv_labels = utils.prepare_example(volume,vox_size,translations[index],category,example)
                    mini_batch.extend(sub_volumes)
                    mb_labels.extend(sv_labels)

            mini_batch,mb_labels = _shuffle(np.array(mini_batch),np.array(mb_labels))
            saveTf(mini_batch,mb_labels,file_counter,mode)
            #print(len(mini_batch))
            file_counter += 1
        except IndexError:
            print("ran out of volumes")
            break
        except KeyboardInterrupt:
            print("keyboard interrupt")
            break

def main_create(mode):
    categories = ["EDGE","LIDC","NEG","POS"] #LIDC 4 and 5 only
    translations = [1,60,1,60]
    if mode == 1:
        categories = ["LIDC","NEG"]
        translations = [20,1]
    total_count = 2500
    vox_size = (32,32,32)
    create_dataset(categories,translations,total_count,vox_size,mode)

if __name__=="__main__":
    main_create(1)

