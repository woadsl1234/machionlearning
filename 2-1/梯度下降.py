import tensorflow as tf
import numpy as np
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2

b=tf.Variable(0.)
k=tf.Variable(0.)
y=k*x_data+b

loss=tf.reduce_mean(tf.square(y_data-y))
optimitizer=tf.train.GradientDescentOptimizer(0.2)
train=optimitizer.minimize(loss)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([k,b]))