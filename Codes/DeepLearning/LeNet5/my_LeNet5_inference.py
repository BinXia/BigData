#coding=UTF-8
import tensorflow as tf

# 配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 32
CONV2_SIZE = 5

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 32
CONV3_SIZE = 5

# 第五层卷积层的尺寸和深度
CONV4_DEEP = 32
CONV4_SIZE = 3

# 第六层卷积层的尺寸和深度
CONV5_DEEP = 32
CONV5_SIZE = 3

# 全连接层的节点格式
FC_SIZE = 512


'''
定义卷积神经网络的前向传播过程
'''
def inference(input_tensor, train, regularizer):
    
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, CONV1_DEEP,CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(relu1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope('layer3-conv3'):
        conv3_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, CONV2_DEEP, CONV3_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(relu2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer4-pool1"):
        pool1 = tf.nn.max_pool(relu3, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")


    with tf.variable_scope('layer5-conv4'):
        conv5_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, CONV3_DEEP,CONV4_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool1, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

    with tf.variable_scope('layer6-conv5'):
        conv6_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, CONV4_DEEP, CONV5_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv6_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        conv6 = tf.nn.conv2d(relu5, conv6_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu6 = tf.nn.relu(tf.nn.bias_add(conv6, conv6_biases))


    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])


    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2 , fc3_weights) + fc3_biases



    return logit