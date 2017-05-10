#coding=UTF-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


'''
定义函数转化变量类型
'''
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def Writer():
    '''
    读取mnist数据
    '''
    mnist = input_data.read_data_sets("../Dataset/MNIST_data",dtype=tf.uint8, one_hot=True)
    
    images = mnist.train.images
    '''
    训练数据所对应的正确答案，可作为一个属性保存在TFRecord中
    '''
    labels = mnist.train.labels
    '''
    训练数据的图像分辨率，这可以作为一Example中的一个属性
    '''
    pixels = images.shape[1]
    num_examples = mnist.train.num_examples

    '''
    输出TFRecord文件的地址
    '''
    filename = "output.tfrecords"
    '''
    创建一个writer来写TFRecord文件
    '''
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        '''
        将图像矩阵转化成一个字符串
        '''
        image_raw = images[index].tostring()

        '''
        将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据格式
        '''
        example = tf.train.Example(features=tf.train.Features(feature={
            'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)
        }))
        '''
        将一个Example写入TFRecord文件
        '''
        writer.write(example.SerializeToString())
    writer.close()


def Reader():
    '''
    创建一个reader来读取TFRecord文件中的样例
    '''
    reader = tf.TFRecordReader()
    '''
    创建一个队列来维护输入文件列表
    '''
    filename_queue = tf.train.string_input_producer(["output.tfrecords"])
    '''
    从文件中读出一个样例，也可以使用read_up_to函数一次读取多个样例
    '''
    _,serialized_example = reader.read(filename_queue)

    '''
    解析读入的样子，如果需要解析多个样例，可以用parse_example函数
    '''
    '''
        TensorFlow提供两种不同的属性解析方法。一种是tf.FixedLenFeature，这种方法得到的解析结果为Tensor，另一种是tf.VarLenFeature，这种方法得到的解析结果为SparseTensor，用于处理稀疏矩阵。
    '''

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'pixels':tf.FixedLenFeature([],tf.int64),
            'label':tf.FixedLenFeature([],tf.int64)
        })

    '''
    tf.decode_raw可以将字符串解析成图像对应的像素数组
    '''
    images = tf.decode_raw(features['image_raw'],tf.uint8)
    labels = tf.cast(features['label'],tf.int32)
    pixels = tf.cast(features['pixels'],tf.int32)

    sess = tf.Session()

    '''
    启动多线程处理输入数据。
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    '''
    每次运行可以读取TFRecord文件中的一个样例，当所有样例读完之后，会从头读取。
    '''
    for i in range(10):
        image, label, pixel = sess.run([images, labels, pixels])




def main():
    Writer()



if __name__ == '__main__':main()









