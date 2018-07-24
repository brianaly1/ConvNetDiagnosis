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

    folder = "ROI"
    if mode==1:
        folder = "NDSB"

    save_path = os.path.join(settings.CA_TRAIN_DATA_DIR,"TFRecords",folder,str(file_index)+".tfrecords")
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
        
def create_dataset(categories,translations,total_count,count_per_tfr,vox_size,mode):
    '''
    Load png files and chunk into mini batches to be saved as TFRecord files

    Inputs:
        categories: list of data categories to use
        translations: list of number of desired translations per category
        total_count: num of examples to be saved in a tf record
        vox_size: desired training set sub volume size        
    '''
    
    base_path = os.path.join(settings.CA_TRAIN_DATA_DIR,"SubVols")

    paths = [os.path.join(base_path,cat) for cat in categories]
    file_names = [os.listdir(path) for path in paths]

    file_counter = 0

    while True:
        try:
            mini_batch = []
            mb_labels = []
            for index,category in enumerate(categories):
                for i in range(0,count_per_tfr[index],translations[index]):
                    example = file_names[index].pop()
                    path = os.path.join(paths[index],example)
                    if category in {"FP","FP2","NDSBNEG1","NDSBPOS1","NDSBNEG2","NDSBPOS2"}:
                        volume = utils.load_cube_img(path, 4, 8, 32)
                    elif category in {"EDGE","NEG","LIDC1","LIDC2","LIDC3","LIDC4","LIDC5","POS"}:
                        volume = utils.load_cube_img(path, 8, 8, 64)
                    sub_volumes,sv_labels = utils.prepare_example(volume,vox_size,translations[index],category,example)
                    mini_batch.extend(sub_volumes)
                    mb_labels.extend(sv_labels)

            mini_batch,mb_labels = _shuffle(np.array(mini_batch),np.array(mb_labels))
            saveTf(mini_batch,mb_labels,file_counter,mode=mode)
            #print(len(mini_batch))
            file_counter += 1
        except IndexError:
            print("ran out of volumes")
            break
        except KeyboardInterrupt:
            print("keyboard interrupt")
            break

def roi_main():
    categories = ["EDGE","LIDC45","NEG","POS","FP","FP2"] 
    translations = [1,80,1,100,1,1]
    total_count = 5000
    count_per_tfr = [960,640,2080,700,260,360] #number of each category included in each tf file
    vox_size = (32,32,32)
    create_dataset(categories,translations,total_count,count_per_tfr,vox_size,mode=0)

def ndsb_main():
    categories = ["NDSBNEG1","NDSBPOS1","NDSBNEG2","NDSBPOS2"]
    translations = [1,1,1,1]
    total_count = 2500
    count_per_tfr = [1340,755,230,175] 
    vox_size = (32,32,32)
    create_dataset(categories,translations,total_count,count_per_tfr,vox_size,mode=1)
    
    

if __name__=="__main__":
    ndsb_main()

