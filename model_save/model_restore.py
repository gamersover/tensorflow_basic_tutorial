#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

meta_path = 'checkpoint/model.ckpt.meta'            #图路径
model_path = 'checkpoint/model.ckpt'                #数据路径
saver = tf.train.import_meta_graph(meta_path)       #加载图

with tf.Session() as sess:
    saver.restore(sess, model_path)                 #加载数据
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('InputData:0')     #加载张量
    y = graph.get_tensor_by_name('LabelData:0')
    accuracy = graph.get_tensor_by_name('Accuracy/acc:0')
    
    #计算测试集的准确度
    test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print('test accuracy', test_acc)