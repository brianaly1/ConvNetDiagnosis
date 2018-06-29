'''
A script to train the c3d network using multiple gpus

Summary of functions:

# function description
output = function_name(inputs)

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import re
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import c3d
import random
import math
import pickle
sys.path.insert(0, '/home/alyb/ConvNetDiagnosis/processing/')
import settings 


MAX_STEPS = 250000
NUM_GPUS = 6
GPUS = ['/gpu:1','/gpu:2','/gpu:3','/gpu:4','/gpu:5','/gpu:6']
BATCH_SIZE = 100
VOL_SHAPE = [32,32,32]
TOT_EXAMPLES = 690000
SHUFFLE_BATCH = 30000
EX_PER_RECORD = 2500
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
    volume = volume / 255
    return volume,label
    
def average_gradients(tower_grads):
    '''
    average gradients for each shared variable accross all towers

    Inputs:
        tower_grads: list of lists of (gradient, variable) tuples - outer list over gradients
                     inner list over towers
    Outputs:
        list of (gradient, variable) averaged across all towers  
    '''
    averaged_grads = []
    for pair_list in zip(*tower_grads):
        grads = []
        for g, _ in pair_list:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = pair_list[0][1]
        pair = (grad, v)
        averaged_grads.append(pair)
    return averaged_grads    
    
def tower_loss(scope,volumes,labels,is_training):
    '''
    Compute the loss for a gpu tower

    Inputs:
        scope: prefix string identifying the tower
        volumes: 5D tensor of shape [batch_size, height, width, depth, 1]
        labels: 1D tensor fof shape [batch_size]
    Outputs:
        A tensor of shape [] containing total loss for input batch  
    '''
    
    # build inference graph
    logits = c3d.inference(volumes,BATCH_SIZE,is_training) 
    
    
    # build loss section of graph and assemble total loss for current tower
    _ = c3d.loss(logits,labels,is_training) # return value would be significant in single gpu training
    
    if is_training == True:
        losses = tf.get_collection('losses',scope)
        total_loss = tf.add_n(losses, name='total_loss')
    else:
        losses = tf.get_collection('val_losses',scope)
        total_loss = tf.add_n(losses, name='val_losses_total')

    all_losses = losses + [total_loss]
    #attach a summary to individual losses and total loss
    for l in all_losses:
        tf.summary.scalar(l.op.name, l)
        
    return total_loss,logits
       
def tower_accuracy(labels,logits):
    '''
    Compute the classification error for the given volumes and labels
    
    Inputs:
        labels: 1D tensor fof shape [batch_size]
        logits: 1D tensor fof shape [batch_size]
    Outputs:
        A tensor of shape [] containing classification error for input batch  
    '''
    predictions = tf.sigmoid(logits)
    predictions = tf.round(predictions)
    predictions = tf.cast(predictions,dtype=tf.int32)
    labels = tf.cast(labels,dtype=tf.int32)
    element_acc = tf.equal(labels,predictions)
    element_acc = tf.cast(element_acc,dtype=tf.float32)
    mean_acc = tf.reduce_mean(element_acc)
    tf.summary.scalar(mean_acc.op.name, mean_acc)
    return mean_acc

def train(train_files,val_files,load_check = False):
    '''
    Train the model for a number of steps
    '''
    train_dir = os.path.join(settings.TRAIN_DATA_DIR,"Checkpoints")
    val_path = os.path.join(settings.TRAIN_DATA_DIR,"Val","validation.p")
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # create a var to count the num of train() calls - number of batches run * num of gpus
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),   
                                       trainable=False)
        # create adam optimizer
        opt = tf.train.AdamOptimizer(c3d.LEARNING_RATE, epsilon = 1e-05)
        
        # inputs from pipeline
        filenames = tf.placeholder(tf.string, shape=[None])
        # is training for batch norm
        is_training = tf.placeholder(tf.bool)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)  # Parse the record into tensors.
        dataset = dataset.shuffle(buffer_size=SHUFFLE_BATCH)
        dataset = dataset.batch(BATCH_SIZE)
        dataset_val = tf.data.TFRecordDataset(val_files)
        dataset_val = dataset_val.map(_parse_function)  # Parse the record into tensors.
        dataset_val = dataset_val.batch(BATCH_SIZE) 
        iterator = dataset.make_initializable_iterator()     
        iterator_val = dataset_val.make_initializable_iterator()

        # calculate gradients per model tower
        tower_grads = []
        # calculate accuracy per model tower
        tower_accs = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i,gpu in enumerate(GPUS):
                with tf.device(gpu):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        # get one batch for the GPU
                        volume_batch, label_batch = iterator.get_next()
                        loss,logits = tower_loss(scope, volume_batch, label_batch, is_training)
                        acc = tower_accuracy(label_batch,logits) 
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        # Calculate the gradients for the batch of data on this tower.
                        grads = opt.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        # Keep track of training accuracies
                        tower_accs.append(acc)
        averaged_grads = average_gradients(tower_grads)
        tower_accs = tf.stack(tower_accs)
        # validate with one gpu
        with tf.variable_scope(tf.get_variable_scope()): 
            with tf.device(GPUS[0]):
                with tf.name_scope('%s' % ('val')) as scope: 
                    tf.get_variable_scope().reuse_variables()
                    val_batch, val_labels = iterator_val.get_next()
                    val_loss,val_logits = tower_loss(scope, val_batch, val_labels, is_training)
                    val_acc = tower_accuracy(val_labels,val_logits)    
        # Add histograms for gradients.
        for grad, var in averaged_grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        # Apply the gradients to the shared variables. 
        apply_gradient_op = opt.apply_gradients(averaged_grads, global_step=global_step)
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
        # Build the summary operation from the last tower summaries.
        #summary_op = tf.summary.merge(summaries)
        summary_op = tf.summary.merge(summaries)
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True))
        sess.run(init)
        sess.run(iterator.initializer, feed_dict={filenames: train_files[0:SHUFFLE_BATCH//EX_PER_RECORD]})
        sess.run(iterator_val.initializer)
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        file_groups = math.ceil(float(TOT_EXAMPLES)/float(SHUFFLE_BATCH)) #chunking the input file list to solve the global shuffle issue
        group_counter = 1 
        epoch_count = 0
        total_accs = []
        moving_avg = 0 
        total_val_accs = []
        total_val_loss = []
        checkpoint_path = os.path.join(train_dir,'model.ckpt')
        if load_check:
            saver.restore(sess, tf.train.latest_checkpoint(train_dir)) 
        for step in xrange(MAX_STEPS):
            start_time = time.time()
            _, loss_value,acc_values,summary_str = sess.run([train_op, loss, tower_accs, summary_op],feed_dict={is_training:True})
            duration = time.time() - start_time
            acc_list = acc_values.tolist()
            total_accs.extend(acc_list)
            for acc in acc_list:
                moving_avg = 0.99*moving_avg + (1-0.99)*acc
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if (step+1) % 10 == 0:
                num_examples_per_step = BATCH_SIZE * NUM_GPUS
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / NUM_GPUS
                mean_acc = np.mean(np.array(total_accs))
                format_str = ('%s: step %d, loss = %.6f, moving avg accuracy = %.4f, avg accuracy = %.4f  (%.1f examples/sec; %.3f '
                      'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, moving_avg, mean_acc, examples_per_sec, sec_per_batch))

            if (step+1) % (int(SHUFFLE_BATCH/(BATCH_SIZE*NUM_GPUS))) == 0:

                summary_writer.add_summary(summary_str, step)

                if group_counter == file_groups:
                    epoch_count = epoch_count + 1
                    moving_avg = total_accs[-1] #reset
                    mean_acc = np.mean(np.array(total_accs))
                    total_accs = [] #reset
                    group_counter = 0 #reset
                    lower_bound = group_counter*SHUFFLE_BATCH//EX_PER_RECORD
                    upper_bound = (group_counter+1)*SHUFFLE_BATCH//EX_PER_RECORD
                    random.shuffle(train_files) 
                    sess.run(iterator.initializer, feed_dict={filenames: train_files[lower_bound:upper_bound]})
                    print("epoch number: %d" %epoch_count)
                    print("epoch average acc: %.4f" %mean_acc)    
                elif group_counter == file_groups-1:
                    lower_bound = group_counter*SHUFFLE_BATCH//EX_PER_RECORD
                    sess.run(iterator.initializer, feed_dict={filenames: train_files[lower_bound:]})
                else:
                    lower_bound = group_counter*SHUFFLE_BATCH//EX_PER_RECORD
                    upper_bound = (group_counter+1)*SHUFFLE_BATCH//EX_PER_RECORD
                    sess.run(iterator.initializer, feed_dict={filenames: train_files[lower_bound:upper_bound]})
                 
                group_counter = group_counter + 1           
                
                # validate
                all_accs = []
                all_losses = []
                print('validating...')
                while True:
                    try:
                        valid_loss,valid_acc = sess.run([val_loss,val_acc],feed_dict={is_training:False})
                        all_accs.append(valid_acc)
                        all_losses.append(valid_loss)
                    except tf.errors.OutOfRangeError:
                        all_accs = np.stack(all_accs)
                        all_losses = np.stack(all_losses)
                        mean_acc = np.mean(all_accs)
                        mean_loss = np.mean(all_losses)
                        total_val_accs.append(mean_acc)
                        total_val_loss.append(mean_loss)
                        print("validation accuracy: %.4f" %mean_acc)
                        print("validation loss: %.4f" %mean_loss)
                        sess.run(iterator_val.initializer)
                        break
                       
            # Save the model checkpoint and validation data.
            if (step+1) % 500 == 0 or step == MAX_STEPS-1:
                saver.save(sess, checkpoint_path, global_step=step)
                with open(val_path,"wb") as openfile:
                    pickle.dump({'accs':total_val_accs,'loss':total_val_loss},openfile) 

def main(argv=None):  # pylint: disable=unused-argument
    data_dir = os.path.join(settings.TRAIN_DATA_DIR,"TFRecords")
    data_files = os.listdir(data_dir)
    training_set = data_files[0:276]
    training_paths = list(map(lambda file_name: os.path.join(data_dir,file_name),training_set))
    validation_set = data_files[276:283]
    validation_paths = list(map(lambda file_name: os.path.join(data_dir,file_name),validation_set))
    train(training_paths,validation_paths,True)


if __name__ == '__main__':
    tf.app.run()    
  
                          
