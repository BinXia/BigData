import tensorflow as tf

'''
1. 生成文件存储样例数据。
'''

'''
创建TFRecord文件的帮助函数
'''
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

'''
模拟海量数据情况下将数据写入不同的文件。
num_shards定义了总共写入多个文件，
instances_per_shard定义了每个文件中多少个数据。
'''
num_shards = 4
instances_per_shard = 500
for i in range(num_shards):
    '''
    将数据分为多个文件时，可以将不同文件以类似0000n-of-0000m的后缀区分。
    其中m表示数据总共被存在了多少个文件中，n表示当前文件的编号。
    式样的方式既方便了通过正则表达式获取文件列表，又在文件名中加入了更多的信息
    '''
    filename = ('data.tfrecords-%.5d-of-%.5d' % (i, num_shards)) 
    '''
    将Example结构写入TFRecord文件。
    '''
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shard):
        '''
        Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
        '''
        example = tf.train.Example(features=tf.train.Features(feature=
            {'i': _int64_feature(i),'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()



'''
2. 读取文件。
'''

'''
使用tf.train.match_filenames_once函数获取文件列表
'''
files = tf.matching_files("data.tfrecords-*")
# files = tf.train.match_filenames_once(tf.gfile.Glob("data.tfrecords-*"))
'''
通过tf.train.string_input_producer函数创建输入队列，输入队列中的文件列表为
tf.train.match_filenames_once函数获取的文件列表。这里讲shuffle参数设为false
来避免随机打乱读文件的顺序。但一般在解决真是问题时，会将shuffle设置为true
'''
filename_queue = tf.train.string_input_producer(files, shuffle=False)
'''
读取并解析一个样本
'''
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
      serialized_example,
      features={
          'i': tf.FixedLenFeature([], tf.int64),
          'j': tf.FixedLenFeature([], tf.int64),
      })

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    '''
    打印文件列表将得到下面的结果：
    '''
    print(sess.run(files))
    '''
    声明tf.train.Coordinator类来协同不同线程，并启动线程。
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    '''
    多次执行获取数据的操作
    '''
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)



'''
3. 组合训练数据（Batching）
'''

'''
这里假设Example结构中i表示一个样例的特征向量，比如一张图像的像素矩阵
而j表示一个样例对应的标签
'''
example, label = features['i'], features['j']
'''
一个batch中样例的个数
'''
batch_size = 2
'''
组合样例的队列中最多可以存储的样例个数。这个队列如果太大，那么需要占用很多内存资源；
如果太小，那么出队操作可能会因为没有数据而被阻碍，从而导致训练效率降低。
一般来说这个队列的大小会和每一个batch的大小相关，下面一行代码给出了设置队列大小的一种方式
'''
capacity = 1000 + 3 * batch_size

'''
使用tf.train.batch函数来组合样例。[example,label]参数给出了需要组合的元素，
一般example和label分别代表训练样本和这个样本对应的正确标签。
batch_size参数给出了每个batch中样例的个数。capacity给出了队列的最大容量。
当队列长度等于容量时，TensorFlow将暂停入队操作，而只是等待元素出队。当元素个数小于容量时，
TensorFlow将自动重新启动入队操作
'''
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    '''
    获取并打印组合之后的样例，在真实问题中，这个输出一般会作为神经网络的输入
    '''
    for i in range(3):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join(threads)








