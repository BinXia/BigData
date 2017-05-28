#coding=UTF-8
import matplotlib.pyplot as plt
import tensorflow as tf   
import numpy as np


image_raw_data = tf.gfile.FastGFile("../Dataset/cat.jpg","rb").read()
'''
1.读取图片
'''
with tf.Session() as sess:
    '''
    将JPEG格式的图像解码得到对应的三维矩阵。
    TensorFlow还提供tf.image.decode_png函数对png格式的图像进行解码。
    解码之后结果为一个张量，在使用它的取值之前需要明确调用运行的过程。
    '''
    img_data = tf.image.decode_jpeg(image_raw_data)
    
    '''
    输出解码之后的三维矩阵
    '''
    print(img_data.eval())

    '''
    获取图像固定范围和通道
    '''
    img_data.set_shape([1797, 2673, 3])
    print(img_data.get_shape())



# '''
# 2.打印图片
# '''
# with tf.Session() as sess:
#   '''
#   使用pyplot工具可视化得到的图像
#   '''
#   plt.imshow(img_data.eval())
#   plt.show()



# '''
# 3.重新调整图片大小
# '''
# with tf.Session() as sess:
#     '''
#     通过tf.image.resize_images函数调整图像的大小。
#     这个函数第一个参数为原始图像，第二个和第三个参数为调整后图像的大小，
#     method参数给出了调整图像大小的算法。
#     '''
#     resized = tf.image.resize_images(img_data, [300, 300], method=0)
    
#     '''
#     TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
#     '''
#     print("Digital type: ", resized.dtype)
#     cat = np.asarray(resized.eval(), dtype='uint8')
#     plt.imshow(cat)
#     plt.show()


# '''
# 4.裁剪和填充图片
# '''
# with tf.Session() as sess:
#     '''
#     通过tf.image.resize_image_with_crop_or_pad函数调整图像的大小。这个函数的
#     第一个参数为原始图像，后面两个参数是调整后的目标图像大小。如果原始图像的尺寸
#     大于目标图像，那么这个函数会自动截取原始图像中居中的部分。如果目标图像尺寸大于
#     原始图像，这个函数会自动在原始图像的四周填充全0背景。因为原始图像的大小为
#     1797*2673，所以下面的第一条命令会自动裁剪，而第二条命令会自动填充。
#     '''
#     croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
#     padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
#     plt.imshow(croped.eval())
#     plt.show()
#     plt.imshow(padded.eval())
#     plt.show()


# '''
# 5.按比例进行截取
# '''
# with tf.Session() as sess:
#     '''
#     通过tf.image.central_crop函数可以按比例裁剪图像。这个函数的第一个参数
#     为原始图像，第二个参数为调整比例，这个比例需要是一个(0,1]的实数
#     '''
#     central_cropped = tf.image.central_crop(img_data, 0.5)
#     plt.imshow(central_cropped.eval())
#     plt.show()


# '''
# 6.翻转图片
# '''
# with tf.Session() as sess: 
#     '''
#     上下翻转
#     '''
#     flipped = tf.image.flip_up_down(img_data)
#     flipped = tf.image.random_flip_up_down(img_data)
#     '''
#     左右翻转
#     '''
#     flipped = tf.image.flip_left_right(img_data)
#     flipped = tf.image.random_flip_left_right(img_data)
#     '''
#     对角线翻转
#     '''
#     transposed = tf.image.transpose_image(img_data)
#     plt.imshow(transposed.eval())
#     plt.show()
    




# '''
# 7.图片色彩调整
# '''
# with tf.Session() as sess:     
#     '''
#     将图片的亮度-0.5
#     '''
#     adjusted = tf.image.adjust_brightness(img_data, -0.5)
    
#     '''
#     将图片的亮度+0.5
#     '''
#     adjusted = tf.image.adjust_brightness(img_data, 0.5)
    
#     '''
#     在[-max_delta, max_delta)的范围随机调整图片的亮度
#     '''
#     adjusted = tf.image.random_brightness(img_data, max_delta=0.5)
    
#     '''
#     将图片的对比度-5
#     '''
#     adjusted = tf.image.adjust_contrast(img_data, -5)
    
#     '''
#     将图片的对比度+5
#     '''
#     adjusted = tf.image.adjust_contrast(img_data, 5)
    
#     '''
#     在[lower, upper]的范围随机调整图的对比度
#     '''
#     adjusted = tf.image.random_contrast(img_data, lower, upper)

#     plt.imshow(adjusted.eval())
#     plt.show()



# '''
# 8.添加色相和饱和度
# '''
# with tf.Session() as sess:
#     '''
#     下面四条命令分别将色相加0.1、0.3、0.6、0.9。
#     '''
#     # adjusted = tf.image.adjust_hue(img_data, 0.1)
#     # adjusted = tf.image.adjust_hue(img_data, 0.3)
#     #adjusted = tf.image.adjust_hue(img_data, 0.6)
#     # adjusted = tf.image.adjust_hue(img_data, 0.9)
    
#     '''
#     在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间
#     '''
#     #adjusted = tf.image.random_hue(image, max_delta)
    
#     '''
#     将图片的饱和度-5。
#     '''
#     #adjusted = tf.image.adjust_saturation(img_data, -5)
#     '''
#     将图片的饱和度+5。
#     '''
#     #adjusted = tf.image.adjust_saturation(img_data, 5)
#     '''
#     在[lower, upper]的范围随机调整图的饱和度。
#     '''
#     #adjusted = tf.image.random_saturation(img_data, lower, upper)
    
#     '''
#     将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
#     '''
#     adjusted = tf.image.per_image_whitening(img_data)
    
#     plt.imshow(adjusted.eval())
#     plt.show()


'''
9.添加标注框并裁剪
'''
with tf.Session() as sess:         
    '''
    设置标注框。一个标注框有四个数字，分别代表[ymin,xmin,ymax,xmax]
    这里给出的是图片的相对位置。
    '''
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    '''
    可以通过提供标注框的方式来告诉随机截取图像的算法哪些部分是“有信息量”的。
    '''
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes)
    '''
    通过标注框可视化随机截取得到的图像
    '''
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    '''
    截取随机出来的图像
    '''
    distorted_image = tf.slice(img_data, begin, size)
    plt.imshow(distorted_image.eval())
    plt.show()