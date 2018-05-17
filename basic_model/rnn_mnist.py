# -*-coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_epochs = 10
batch_size = 64
n_input = 784
time_steps = 28
input_size = 28
num_classes = 10
rnn_size = 128
lr = 0.01

x = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.float32, shape=[None, num_classes])

def rnn_model(x):
    x = tf.reshape(x, shape=[-1, time_steps, input_size])
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    print(states)
    output = tf.transpose(outputs, [1,0,2])[-1]
    fc_w = tf.Variable(tf.random_normal([rnn_size, num_classes]))
    fc_b = tf.Variable(tf.random_normal([num_classes]))
    return tf.matmul(output, fc_w) + fc_b

logits = rnn_model(x)
prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    total_batch = mnist.train.num_examples // batch_size
    for epoch in range(train_epochs):
        for batch in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x:batch_x, y:batch_y})
            
            if batch % 200 == 0:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={x:batch_x, y:batch_y})
                print("epoch {}, batch {}, loss {:.4f}, accuracy {:.3f}".format(epoch, batch, loss, acc))
        
    print("optimization finished")
    
    test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print("test acc", test_acc)
