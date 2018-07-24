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
import pandas
import c3d_train

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

def predict(files): #load graph...

    CHECKPOINT = os.path.join(settings.CA_TRAIN_DATA_DIR,"Checkpoints","roi")

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        filenames = tf.placeholder(tf.string, shape=[None])

        # for dropout
        keep_prob = tf.placeholder(dtype=tf.float32)

        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_initializable_iterator() 

        with tf.variable_scope(tf.get_variable_scope()):  
            with tf.device(GPUS[0]):
                with tf.name_scope('infer') as scope: 
                    volume_batch,labels,centers = iterator.get_next()
                    logits,_ = c3d.inference(volume_batch,BATCH_SIZE,tf.constant(False,dtype=tf.bool),keep_prob=keep_prob)  
                    predictions = logits    
                    predictions = tf.sigmoid(logits)

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
                pred_vals,label_vals,vol_vals,cen_vals = sess.run([predictions,labels,volume_batch,centers],feed_dict={keep_prob:1.0})
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

def test_roi(test_paths):

    fps = [] 
    fns = []
    tps = []
    tns = []

    file_counter = 0

    for path in test_paths:

        uid = path.split('/')[-1][:-10]
        cat_names = ["FPS","FNS","TPS","NODS"]
        res_dir = os.path.join(settings.CA_TEST_RES_DIR,"LUNA",uid)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
        sub_dirs = [os.path.join(res_dir,x) + "/" for x in cat_names[0:-1]]
        [os.mkdir(sub_dir) for sub_dir in sub_dirs if not os.path.exists(sub_dir)]
        
        counters = [0,0,0]

        fps.append([])
        fns.append([])
        tps.append([])
        tns.append([])

        predictions_np,labels_np,volumes_np,cens_np = predict([path])
        predictions_np[predictions_np > settings.THRESHOLD] = 1
        predictions_np[predictions_np <= settings.THRESHOLD] = 0
        predictions_np = predictions_np.astype(np.int32)
        acc = np.mean((predictions_np == labels_np).astype(np.float32)) 
        for index in range(np.shape(labels_np)[0]):
            cen = cens_np[index].tolist()
            volume = np.squeeze(volumes_np[index])
            if predictions_np[index] == 1 and labels_np[index]==0:
                target_path = os.path.join(sub_dirs[0],uid + "_" + str(counters[0]) + "_0_" + "fp.png")
                fps[-1].append(cen)
                utils.save_cube_img(target_path, volume*255, 4, 8)
                counters[0]+=1
            elif predictions_np[index] == 0 and labels_np[index]==1:
                target_path = os.path.join(sub_dirs[1],uid + "_" + str(counters[1]) + "_1_" + "fn.png")
                target_path = os.path.join(settings.CA_TRAIN_DATA_DIR,"FN",uid+"_"+str(counters[1]) + "_1_" + "fn.png")
                fns[-1].append(cen)
                utils.save_cube_img(target_path, volume*255, 4, 8)
                counters[1]+=1
            elif predictions_np[index] == 1 and labels_np[index]==1:
                target_path = os.path.join(sub_dirs[2],uid + "_" + str(counters[1]) + "_1_" + "tp.png")
                tps[-1].append(cen)
                utils.save_cube_img(target_path, volume*255, 4, 8)
                counters[2]+=1
            elif predictions_np[index] == 0 and labels_np[index]==0:
                tns[-1].append(cen)
        
        file_counter += 1
        
        print("{} --- acc is: {}, fps: {}, fns: {},tps: {},tns: {}".format(file_counter,acc,len(fps[-1]),len(fns[-1]),len(tps[-1]),len(tns[-1])))
        print("TP locations are: ")
        print(tps[-1])
        print("FP locations are: ")
        print(fps[-1])
        print("FN locations are: ")
        print(fns[-1])
        nods = [utils.get_patient_nodules(uid)]
        print("Nodule locations:: {}".format(nods))
        print("------------------------------------------------------------------------------------------")

        for ind,cat in enumerate([fps[-1],fns[-1],tps[-1],nods[-1]]):
            lines = [[index,cen[0],cen[1],cen[2]] for index,cen in enumerate(cat)]
            df_annos = pandas.DataFrame(lines, columns=["anno_index", "coord_x", "coord_y", "coord_z"])
            save_path = os.path.join(res_dir,uid + "_" + cat_names[ind] + "_annos.csv")
            df_annos.to_csv(save_path, index=False)
        visualize.view_candidates(path,fps[-1])


def test_cancer(files,mag=1):

    cat_names = ["NDSBNEG"+str(mag),"NDSBPOS"+str(mag)] 

    save_dirs = [os.path.join(settings.CA_TRAIN_DATA_DIR,cat) for cat in cat_names]
    [os.mkdir(dir_name) for dir_name in save_dirs if not os.path.exists(dir_name)]

    file_count = 0
    count = 0

    for path in files:
        
        try:
            uid = path.split('/')[-1][:-10]

            predictions_np,labels_np,volumes_np,cens_np = predict([path])
            predictions_np[predictions_np > settings.THRESHOLD] = 1
            predictions_np[predictions_np <= settings.THRESHOLD] = 0
            predictions_np = predictions_np.astype(np.int32)

            pos_pred = np.nonzero(predictions_np)
            pos_vols = volumes_np[pos_pred]                #use centroids and code util func to extract vol from cen - so that can have 64^3 so can augment via trans
            corr_labels = labels_np[pos_pred]
            save_dir = save_dirs[corr_labels[0]]
            [utils.save_cube_img(os.path.join(save_dir,uid + "_" + str(index) + "_.png"), np.squeeze(volume)*255, 4, 8) for index,volume in enumerate(pos_vols)]

            file_count+=1
            count+=np.shape(pos_vols)[0]
            print("{} : {} : {} label and {} positive sub volumes".format(file_count,uid,corr_labels[0],np.shape(pos_vols)[0]))

        except Exception as e:
            print("Exception: {}".format(e))

    print("Total count: {}".format(count))
                   
def nsdb_main(mag):
    data_dir = os.path.join(settings.CA_TRAIN_DATA_DIR,"TFRecordsTest","NDSB",str(mag))
    tf_data_files = os.listdir(data_dir)  
    tf_test_set = tf_data_files
    tf_test_paths = list(map(lambda file_name: os.path.join(data_dir,file_name),tf_test_set))
    test_cancer(tf_test_paths,mag=mag) 
    

if __name__=="__main__":
    ndsb_main(mag=2)    
