# -*-coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_epochs = 10
batch_size = 64
n_input = 784     # 图像大小
time_steps = 28   # 时间步长
input_size = 28   # 序列长度
num_classes = 10
rnn_size = 128    # rnn隐藏层大小
lr = 0.01

# 定义输入
x = tf.placeholder(tf.float32, shape=[None, n_input])
y = tf.placeholder(tf.float32, shape=[None, num_classes])

# 定义网络结构
def rnn_model(x):
    # 将输入x变为[batch_size, time_steps, input_size]
    x = tf.reshape(x, shape=[-1, time_steps, input_size])
    # 构建rnn
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    # 将输入送入rnn，得到输出与中间状态，输出shape为[batch_size, time_steps, rnn_size]
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    # 获取最后一个时刻的输出，输出shape为[batch_size, rnn_size]
    output = tf.transpose(outputs, [1,0,2])[-1]
    # 全连接层，最终输出大小为[batch_size, num_classes]
    fc_w = tf.Variable(tf.random_normal([rnn_size, num_classes]))
    fc_b = tf.Variable(tf.random_normal([num_classes]))
    return tf.matmul(output, fc_w) + fc_b

# 构建网络
logits = rnn_model(x)
prediction = tf.nn.softmax(logits)

# 定义损失函数与优化器
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)

# 定义评价指标
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 开始训练网络
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
    
    # 测试
    test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print("test acc", test_acc)
