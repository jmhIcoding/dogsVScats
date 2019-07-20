#coding:utf-8
#将kaggle的图片转换为tf_record格式的文件
__author__ = 'jmh081701'
import numpy as np
import tensorflow as tf
import sys
import  os
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_example(filename, image_buffer, label, text, height, width):
    colorspace = 'RGB'.encode()
    channels = 3
    image_format = 'JPEG'.encode()
    if not isinstance(label,int):
        label=int(label)
    if not isinstance(text,bytes):
        text = text.encode()
    if not isinstance(filename,bytes):
        filename = filename.encode()
    example = tf.train.Example(features=tf.train.Features(
        feature={
                    'image/height': _int64_feature(height), #图片高度
                    'image/width': _int64_feature(width),   #图片宽度
                    'image/colorspace': _bytes_feature(colorspace),
                    'image/channels': _int64_feature(channels),#通道个数
                    'image/class/label': _int64_feature(label),#label
                    'image/class/text': _bytes_feature(text),
                    'image/format': _bytes_feature(image_format),#'JPEG'
                    'image/filename': _bytes_feature(os.path.basename(filename)), #文件名
                    'image/encoded': _bytes_feature(image_buffer)#图片的内容
                }
                )
                )
    return example

def convert_kaggle_image(datadir,usage='train'):

    _decode_jpeg_data = tf.placeholder(dtype=tf.string)#place holder
    _decode_jpeg = tf.image.decode_jpeg(_decode_jpeg_data, channels=3)

    with tf.Session() as sess:
        for root,subdirs,files in os.walk(datadir):
            counter = 0
            shard   = 5
            each_shard=int(len(files)/shard)
            writers=[0,1,2,3,4]
            writers[0] = tf.python_io.TFRecordWriter('.\\dogsVScats_%s_0-of-5.tfrecord'%usage)
            writers[1] = tf.python_io.TFRecordWriter('.\\dogsVScats_%s_1-of-5.tfrecord'%usage)
            writers[2] = tf.python_io.TFRecordWriter('.\\dogsVScats_%s_2-of-5.tfrecord'%usage)
            writers[3] = tf.python_io.TFRecordWriter('.\\dogsVScats_%s_3-of-5.tfrecord'%usage)
            writers[4] = tf.python_io.TFRecordWriter('.\\dogsVScats_%s_4-of-5.tfrecord'%usage)
            for file in files:
                writer = writers[int(counter/each_shard)]
                label = 0 if file.split('.')[0] == 'cat' else 1  #0 is cat,while 1 is dog
                filename=root+"\\"+file
                # Read the image file, mode:read and binary
                image_data_raw=tf.gfile.GFile(filename,"rb").read()
                # Convert 2 tensor,转换的目的是为了提取height和width,也可以使用PIL库来转换
                image= sess.run(_decode_jpeg,feed_dict={_decode_jpeg_data:image_data_raw})
                height=image.shape[0]
                width =image.shape[1]

                example=_convert_example(filename,image_buffer=image_data_raw,label=label,text="",height=height,width=width)
                writer.write(example.SerializeToString())
                counter+=1
                print("Finish:%s"%str(counter/len(files)))
                sys.stdout.flush()
            writers[0].close()
            writers[1].close()
            writers[2].close()
            writers[3].close()
            writers[4].close()
if __name__ == '__main__':
    convert_kaggle_image(datadir=r"G:\bdndisk\kaggle\train\validation",usage='validation')
    convert_kaggle_image(datadir=r"G:\bdndisk\kaggle\train\train",usage='train')
