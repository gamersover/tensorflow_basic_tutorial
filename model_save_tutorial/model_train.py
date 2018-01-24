# coding: utf-8
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

learning_rate = 0.001
train_epochs = 10
batch_size = 64
checkpoint_path = 'checkpoint/'

n_input = 784
n_hidden1 = 100
n_hidden2 = 100
n_classes = 10

#name参数，记录变量名字
x = tf.placeholder(tf.float32, shape=[None, n_input], name='InputData')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='LabelData')

weights = {'w1': tf.Variable(tf.random_normal([n_input, n_hidden1]), name='W1'),
                  'w2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2]), name='W2'),
                  'w3': tf.Variable(tf.random_normal([n_hidden2, n_classes]), name='W3')}
biases = {'b1': tf.Variable(tf.random_normal([n_hidden1]), name='b1'),
                'b2': tf.Variable(tf.random_normal([n_hidden2]), name='b2'),
                'b3': tf.Variable(tf.random_normal([n_classes]), name='b3')}

def inference(input_x):
    layer_1 = tf.nn.relu(tf.matmul(x, weights['w1']) + biases['b1'])
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['w2']) + biases['b2'])
    out_layer = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return out_layer

#定义计算过程的名字
with tf.name_scope('Inference'):
    logits = inference(x)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope('Accuracy'):
    pre_correct = tf.equal(tf.argmax(y, 1), tf.argmax(tf.nn.softmax(logits), 1))
    accuracy = tf.reduce_mean(tf.cast(pre_correct, tf.float32), name='acc')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)
    
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)  #获取checkpoint状态
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint_path+'model.ckpt')        #加载数据
        print('continue last train!!')
    else:
        print('restart train!!')
    for epoch in range(train_epochs):
        for batch in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x:batch_x, y:batch_y})   
        if (epoch+1) % 5 == 0:
            loss_, acc = sess.run([loss, accuracy], feed_dict={x:batch_x, y:batch_y})
            print("epoch {},  loss {:.4f}, acc {:.3f}".format(epoch, loss_, acc))
            saver.save(sess, checkpoint_path+'model.ckpt')    #模型名字model.ckpt
        
    print("optimizer finished!")
    print("模型保存在"， checkpoint_path)