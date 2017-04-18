#coding=UTF8
import tensorflow as tf
import numpy as np


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



'''
定义训练数据batch的大小
'''
batch_size = 8


'''
定义神经网络的参数
'''
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
b1 = tf.Variable(tf.zeros([3]))
variable_summaries('w1',w1)
variable_summaries('b1',b1)
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
b2 = tf.Variable(tf.zeros([1]))
variable_summaries('w2',w2)
variable_summaries('b2',b2)

'''
在shape的一个维度上使用None可以方便使用不大的batch大小。
在训练时需要把数据分成比较小的batch，但是在测试时，可以一次性使用全部的数据。
当数据集比较小时，这样比较方便测试，但是数据集比较大时，将大量数据放入一个batch可能会导致内存溢出。
'''
with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
	y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-input')

'''
定义神经网络前向传播的过程
'''
a = tf.matmul(x,w1)
# a = tf.add(tf.matmul(x,w1),b1)
y = tf.matmul(a,w2)
# y = tf.add(tf.matmul(a,w2),b2)

'''
定义损失函数和反向传播的算法
'''
with tf.name_scope('loss_function'):
	cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
with tf.name_scope('train_step'):
	train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


'''
生成一个写日志的LOG，并将当前的TensorFlow计算图写入日志
'''
LOG = tf.summary.FileWriter("./log",tf.get_default_graph())
monitor = tf.summary.merge_all()

'''
通过随机数生成一个模拟数据集
'''
rdm = np.random.RandomState(1)
dataset_size = 128
INPUT = rdm.rand(dataset_size, 2)

'''
定义规则来给出样本的标签。在这里，所有x1和x2<1的样例都会被认为是正样本（比如零件合格），
而其他为负样本（比如零件不合格）。
'''
LABEL = [[int(x1+x2 < 1)] for (x1,x2) in INPUT]



'''
创建一个会话来运行TensorFlow程序
'''
with tf.Session() as sess:
	'''
	初始化变量
	'''
	init_op = tf.initialize_all_variables()
	sess.run(init_op)

	'''
	在训练之前神经网络参数的值
	'''
	print(sess.run(w1))
	print(sess.run(w2))

	'''
	设定迭代的次数
	'''
	epoch = 5000
	for i in range(epoch):
		'''
		每次选取batch_size个样本进行训练
		'''
		start = (i*batch_size)%dataset_size
		end = min(start+batch_size,dataset_size)

		'''
		通过选取的样本训练神经网络并更新参数
		'''
		summary,_ = sess.run([monitor,train_step],feed_dict={x:INPUT,y_:LABEL})
		LOG.add_summary(summary,i)
		'''
		每隔一段时间计算在所有数据上的交叉熵并输出
		'''
		if i % 1000 == 0:
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
			total_cross_entropy = sess.run(cross_entropy,feed_dict={x:INPUT,y_:LABEL},options=run_options,run_metadata=run_metadata)
			'''
			将节点在运行时的信息写入日志文件
			'''
			LOG.add_run_metadata(run_metadata, 'step%03d'%i)
			print("After %d training step(s), cross entropy on all data data is %g"%(i, total_cross_entropy))

	'''
	在训练之后的神经网络参数的值
	'''
	print(sess.run(w1))
	print(sess.run(w2))

LOG.close()
		