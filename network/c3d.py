import tensorflow as tf
import numpy as np


NUM_CLASSES = 1
VOLUME_SIZE = [32,32,32] # z,y,x
LEARNING_RATE = 0.00001 #
KEEP_RATE = 0.6

def conv3d(x, W,sk,sj,si):
    return tf.nn.conv3d(x, W, strides=[1,sk,sj,si,1], padding='SAME')

def max_pool3d(x,k,j,i,name):
    return tf.nn.max_pool3d(x, ksize=[1,k,j,i,1], strides=[1,k,j,i,1], padding='SAME', name=name)

def avg_pool3d(x,k,name):
    return tf.nn.avg_pool3d(x, ksize=[1,k,1,1,1], strides=[1,k,1,1,1], padding='SAME', name=name)

def activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations',x)
  tf.summary.scalar(tensor_name + '/sparsity',tf.nn.zero_fraction(x))

def variable_on_cpu(name, shape, initializer):
    """
    Inputs:
        name: string
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def variable_with_weight_decay(name, shape, wd, is_training):
    """
    Weight decay is added only if one is specified.

    Inputs:
        name: string
        shape: list of ints
        stddev: for truncated Gaussian
        wd: L2 loss
    Returns:
        Variable Tensor
    """
    dtype = tf.float32
    var = variable_on_cpu(name,shape, tf.contrib.layers.xavier_initializer(dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        if is_training == True:
            tf.add_to_collection('losses', weight_decay)
        else:
            tf.add_to_collection('val_losses', weight_decay)

    return var

def inference(volumes,batch_size,is_training):
    '''
    Perform a forward pass and return output logit/s

    Inputs:
        volumes: 5D tensor of shape [batch_size, width, height, depth, channels]
    Outputs:
        sigmoid: class prediction
    '''
 
    # downsample Z
    downsampled = avg_pool3d(volumes,k=2,name='avgpool1')
    # first layer
    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weights',[3,3,3,1,64],0.0005,is_training)
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = conv3d(downsampled, kernel,1,1,1) + biases
        #pre_activation = tf.contrib.layers.batch_norm(pre_activation,is_training=is_training)
        conv1 = tf.nn.leaky_relu(pre_activation,name='conv1')
        activation_summary(conv1)
    pool1 = max_pool3d(conv1,1,2,2,name='pool1')

    # second layer
    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights',[3,3,3,64,128],0.0005,is_training)
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = conv3d(pool1, kernel,1,1,1) + biases
        #pre_activation = tf.contrib.layers.batch_norm(pre_activation,is_training=is_training)
        conv2 = tf.nn.leaky_relu(pre_activation,name='conv2')
        activation_summary(conv2)
    pool2 = max_pool3d(conv2,2,2,2,name='pool2')

    # third layer
    with tf.variable_scope('conv3a') as scope:
        kernel = variable_with_weight_decay('weights',[3,3,3,128,256],0.0005,is_training)
        biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        pre_activation = conv3d(pool2, kernel,1,1,1) + biases
        #pre_activation = tf.contrib.layers.batch_norm(pre_activation,is_training=is_training)
        conv3a = tf.nn.leaky_relu(pre_activation,name='conv3a')
        activation_summary(conv3a)

    with tf.variable_scope('conv3b') as scope:
        kernel = variable_with_weight_decay('weights',[3,3,3,256,256],0.0005,is_training)
        biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
        pre_activation = conv3d(conv3a, kernel,1,1,1) + biases
        #pre_activation = tf.contrib.layers.batch_norm(pre_activation,is_training=is_training)
        conv3b = tf.nn.leaky_relu(pre_activation,name='conv3b')
        activation_summary(conv3b) 
    pool3 = max_pool3d(conv3b,2,2,2,name='pool3')

    # fourth layer
    with tf.variable_scope('conv4a') as scope:
        kernel = variable_with_weight_decay('weights',[3,3,3,256,512],0.0005,is_training)
        biases = variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = conv3d(pool3, kernel,1,1,1) + biases
        #pre_activation = tf.contrib.layers.batch_norm(pre_activation,is_training=is_training)
        conv4a = tf.nn.leaky_relu(pre_activation,name='conv4a')
        activation_summary(conv4a)

    with tf.variable_scope('conv4b') as scope:
        kernel = variable_with_weight_decay('weights',[3,3,3,512,512],0.0005,is_training)
        biases = variable_on_cpu('biases', [512], tf.constant_initializer(0.0))
        pre_activation = conv3d(conv4a, kernel,1,1,1) + biases
        #pre_activation = tf.contrib.layers.batch_norm(pre_activation,is_training=is_training)
        conv4b = tf.nn.leaky_relu(pre_activation,name='conv4b')
        activation_summary(conv4b) 
    pool4 = max_pool3d(conv4b,2,2,2,name='pool4')
    # bottleneck layer
    with tf.variable_scope('bottleneck') as scope:
        kernel = variable_with_weight_decay('weights',[2,2,2,512,64],0.0005,is_training)
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = conv3d(pool4, kernel,2,2,2) + biases
        pre_dropout = tf.nn.leaky_relu(pre_activation,name='bottle_neck_dropout')
        bottle_neck = tf.nn.dropout(pre_dropout,KEEP_RATE,name='bottle_neck')
        activation_summary(bottle_neck) 
  
    # output layer
    with tf.variable_scope('logistic') as scope:
        kernel = variable_with_weight_decay('weights',[1,1,1,64,1],0.0005,is_training)
        biases = variable_on_cpu('biases', [1], tf.constant_initializer(0.0))
        pre_activation = conv3d(bottle_neck, kernel,1,1,1) + biases
        logits = tf.squeeze(pre_activation,[2,3,4],name='logits')  
        activation_summary(logits)

    return logits


def loss(logits, labels, is_training):
    """
    compute cross entropy loss and l2 loss of all weight decay vars
    Inputs:
        logits: output of inference().
        labels: 1-D tensor of shape [batch_size]
    Outputs:
        Loss tensor
    """
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    if is_training == True:
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
    else:
        tf.add_to_collection('val_losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('val_losses'), name='val_loss_total')

    

