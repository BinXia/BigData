#! /usr/bin/python2.7
#-*- encoding:utf-8 -*-
from datetime import datetime
import math
import time
import tensorflow as tf
import my_AlexNet_inference
from tensorflow.examples.tutorials.mnist import input_data


# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 600
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH="../Models/MNIST_1/"
MODEL_NAME="AlexNet_model_1"


def train(mnist):
    # 定义输入输出placeholder
    '''
    x = tf.placeholder(tf.float32, [None, my_AlexNet_inference.INPUT_NODE], name='x-input')
    '''
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,                           # 第一维表示一个batch中样例的个数
            my_AlexNet_inference.IMAGE_SIZE,    # 第二维和第三维表示图片的尺寸
            my_AlexNet_inference.IMAGE_SIZE,
            my_AlexNet_inference.NUM_CHANNELS   # 第四维表示图片的深度，对于RBG格式的图片，深度为5
        ], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, my_AlexNet_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 直接使用my_AlexNet_inference.py中定义的前向传播过程
    y = my_AlexNet_inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

def main(argv=None):
    mnist = input_data.read_data_sets("../Dataset/MNIST_data_1", one_hot=True)
    train(mnist)

if __name__ == '__main__':tf.app.run()


