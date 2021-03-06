#coding=UTF8
import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None


def variable_summaries(name, var):
  '''
  通过tf.histogram_summary函数记录张量中元素的取值分布。对于给出的图表名称和张量，
  tf.histogram_summary函数会生成一个Summary protocol buffer。
  将Summary写入TensorBoard日志文件后，可以在HISTOGRAMS界面看到对应名称的图表。
  '''
  tf.summary.histogram(name,var)

  '''
  计算变量的平均值，并定义生成平均值信息日志的操作。
  '''
  mean = tf.reduce_mean(var)
  tf.summary.scalar('mean/'+name, mean)
  '''
  计算变量的标准差，并定义生成其日志的操作。
  '''
  stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
  tf.summary.scalar('stddev/'+name, stddev)



def main(_):
  '''
  导入数据
  '''
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  '''
  在shape的一个维度上使用None可以方便使用不大的batch大小。
  在训练时需要把数据分成比较小的batch，但是在测试时，可以一次性使用全部的数据。
  当数据集比较小时，这样比较方便测试，但是数据集比较大时，将大量数据放入一个batch可能会导致内存溢出。
  '''
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')


  '''
  定义神经网络的参数
  '''
  w1 = tf.Variable(tf.random_normal([784, 100], stddev=1))
  b1 = tf.Variable(tf.zeros([100]))
  variable_summaries('w1',w1)
  variable_summaries('b1',b1)
  w2 = tf.Variable(tf.random_normal([100,10], stddev=1))
  b2 = tf.Variable(tf.zeros([10]))
  variable_summaries('w2',w2)
  variable_summaries('b2',b2)

  '''
  定义神经网络前向传播的过程
  '''
  # a = tf.matmul(x,w1)
  a = tf.add(tf.matmul(x,w1),b1)
  # y = tf.matmul(a,w2)
  y = tf.add(tf.matmul(a,w2),b2)

  '''
  定义损失函数和反向传播的算法
  '''
  with tf.name_scope('loss_function'):
    cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)),name='Corss-Entropy')
  with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

  '''
  定义神经网络模型的评价指标
  '''
  with tf.name_scope('criteria'):
    '''
    分类正确率
    '''
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)

  '''
  生成一个写日志的LOG，并将当前的TensorFlow计算图写入日志
  '''
  LOG = tf.summary.FileWriter("./log",tf.get_default_graph())
  monitor = tf.summary.merge_all()
  
  '''
  创建一个会话来运行TensorFlow程序
  '''
  with tf.Session() as sess:
    '''
    初始化变量
    '''
    tf.global_variables_initializer().run()

    '''
    设定迭代的次数
    '''
    epoch = 1000
    for i in xrange(epoch):
      '''
      每次选取100个样本进行训练
      '''
      batch_xs, batch_ys = mnist.train.next_batch(100)
      '''
      通过选取的样本训练神经网络并更新参数
      '''
      summary,_ = sess.run([monitor,train_step],feed_dict={x:batch_xs,y_:batch_ys})
      LOG.add_summary(summary,i)

      if i % 100 == 0:
        '''
        配置运行时需要记录的信息
        '''
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        '''
        运行时记录运行信息的proto
        '''
        run_metadata = tf.RunMetadata()
        '''
        将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间、空间开销
        '''
        cross_entropy_train = sess.run(cross_entropy,feed_dict={x:batch_xs,y_:batch_ys},options=run_options,run_metadata=run_metadata)
        acc_test = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
        '''
        将节点在运行时的信息写入日志文件
        '''
        LOG.add_run_metadata(run_metadata, 'step%03d'%i)
        print "After %d training step(s), cross entropy on training data is %g and accuracy on test data is %g."%(i, cross_entropy_train,acc_test)

  LOG.close()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='../Dataset',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    