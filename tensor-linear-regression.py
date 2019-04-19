# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:06:32 2019

This is simple linear regression

@author: xqiu3
"""

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

# -------------------------- Generate random linear data

# data size
n = 50

x = np.linspace(0,50,n)
y = np.linspace(0,50,n)

# add noise to x, y
x += np.random.uniform(-5,5,n)
y += np.random.uniform(-5,5,n)

# plot training data
plt.scatter(x,y,color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#---------------------------- build optimizer

learning_rate = 0.01
training_epochs = 1000

# create placeholder for test data 
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Declare variable
W = tf.Variable(np.random.rand(),name = "W")
b = tf.Variable(np.random.rand(),name = "b")

# Hypothesis
y_pred = tf.add(tf.multiply(X,W),b)

# Cost function
cost = tf.reduce_sum(tf.pow(y_pred - Y,2))/ (2 * n)

# Use gradient descent to minimize the cost. this is a process of finding best W and b
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# S----------------------------tarting the Tensorflow Session
# Global Variables Initializer 
init = tf.global_variables_initializer()

# start feeding data
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (_x,_y) in zip(x,y):
            sess.run(optimizer,feed_dict = {X:_x,Y:_y})
          
# Storing necessary values to be used outside the Session 
    weight = sess.run(W) 
    bias = sess.run(b) 

# Calculating the predictions 
predictions = weight * x + bias 


# plot
plt.plot(x, y, 'ro', label ='Original data') 
plt.plot(x, predictions, label ='Fitted line') 
plt.title('Linear Regression Result') 
plt.legend() 
plt.show()     



