#-*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf

learning_rate = 0.001
train_epochs = 1
batch_size = 64
n_input = 784
n_classes = 10

x = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)

#权重的形状为[kernel_size, kernel_size, in_channels, out_channels]
#-------------卷积核高-------卷积核宽-----输入通道数----输出通道数--
weights = {'wc1': tf.Variable(tf.random_normal([5,5,1,20])),
           'wc2': tf.Variable(tf.random_normal([5,5,20,50])),
           'wf1': tf.Variable(tf.random_normal([4*4*50, 500])),
           'wf2': tf.Variable(tf.random_normal([500, 10]))}

biases = {'bc1': tf.Variable(tf.random_normal([20])),
          'bc2': tf.Variable(tf.random_normal([50])),
          'bf1': tf.Variable(tf.random_normal([500])),
          'bf2': tf.Variable(tf.random_normal([10]))}

def inference(x):
	#将图片大小变为[batch_size, height, width, channels]
	#---------------训练个数------高-----宽------通道---
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    #步长stride中间两个维度表示高和宽，其他两个维度默认为1即可
    #卷积层C1
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, biases['bc1'])
    #池化层P1
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding='VALID')
    
    #卷积层C2
    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, biases['bc2'])
    #池化层P2
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],  padding='VALID')
   
    #将4*4*50变为800
    fc1 = tf.reshape(conv2, [-1, weights['wf1'].get_shape().as_list()[0]])
    #全连接层F1
    fc1 = tf.nn.xw_plus_b(fc1, weights['wf1'], biases['bf1'])
    fc1 = tf.nn.relu(fc1)
    #dropout层, dropout原理参考https://yq.aliyun.com/articles/68901
    fc1 = tf.nn.dropout(fc1, keep_prob)
    #全连接层F2(输出)
    out = tf.nn.xw_plus_b(fc1, weights['wf2'], biases['bf2'])
    return out


logits = inference(x)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

pre_correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(pre_correct, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)
    for epoch in range(train_epochs):
        for batch in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x:batch_x, y:batch_y, keep_prob:0.8})

            if batch % 80 == 0:
	            loss, acc = sess.run([loss_op, accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:1.0})
	            print("epoch {}, batch {}, loss {:.4f},  accuracy {:.3f}".format(epoch, batch, loss, acc))

    print("optimization finished!")
    
    #在测试集上测试
    test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
    print('test accuracy', test_acc)