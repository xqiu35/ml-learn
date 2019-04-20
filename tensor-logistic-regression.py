# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:38:32 2019

@author: xqiu3
"""
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder 


# Parameters
# Parameters
alpha = 0.25
epochs = 50


# prepare data
df = pd.read_csv('ex2data1.txt', header=None) # no header
# Feature Matrix 
x_orig = df.iloc[:, 0:-1].values 
  
# Data labels 
y_orig = df.iloc[:, -1:].values 

# Positive Data Points 
x_pos = np.array([x_orig[i] for i in range(len(x_orig)) 
                                    if y_orig[i] == 1]) 

# Negative Data Points 
x_neg = np.array([x_orig[i] for i in range(len(x_orig)) 
                                    if y_orig[i] == 0]) 
  
# Plotting the Positive Data Points 
plt.scatter(x_pos[:, 0], x_pos[:, 1], color = 'blue', label = 'Positive') 
plt.scatter(x_neg[:, 0], x_neg[:, 1], color = 'red', label = 'Negative')
plt.show()

###############################################################################################
oneHot = OneHotEncoder() 
oneHot.fit(x_orig) 
x = oneHot.transform(x_orig).toarray() 
oneHot.fit(y_orig) 
y = oneHot.transform(y_orig).toarray() 


# tf Graph Input
m, n = x.shape 

X = tf.placeholder(tf.float32, [None, n])
Y = tf.placeholder(tf.float32, [None, 2]) 

# Set model weights
W = tf.Variable(tf.zeros([n, 2]))
B = tf.Variable(tf.zeros([2]))

# Construct model
Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), B)) 

# Sigmoid Cross Entropy Cost Function 
cost = tf.nn.sigmoid_cross_entropy_with_logits( 
                    logits = Y_hat, labels = Y) 
  
# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer( 
         learning_rate = alpha).minimize(cost) 

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

######################### Starting the Tensorflow Session #######################################
# Starting the Tensorflow Session 
with tf.Session() as sess: 
	
	# Initializing the Variables 
	sess.run(init) 
	
	# Lists for storing the changing Cost and Accuracy in every Epoch 
	cost_history, accuracy_history = [], [] 
	
	# Iterating through all the epochs 
	for epoch in range(epochs): 
		cost_per_epoch = 0
		
		# Running the Optimizer 
		sess.run(optimizer, feed_dict = {X : x, Y : y}) 
		
		# Calculating cost on current Epoch 
		c = sess.run(cost, feed_dict = {X : x, Y : y}) 
		
		# Calculating accuracy on current Epoch 
		correct_prediction = tf.equal(tf.argmax(Y_hat, 1), 
										tf.argmax(Y, 1)) 
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, 
												tf.float32)) 
		
		# Storing Cost and Accuracy to the history 
		cost_history.append(sum(sum(c))) 
		accuracy_history.append(accuracy.eval({X : x, Y : y}) * 100) 
		
		# Displaying result on current Epoch 
		if epoch % 2 == 0 and epoch != 0: 
			print("Epoch " + str(epoch) + " Cost: "
							+ str(cost_history[-1])) 
	
	Weight = sess.run(W) # Optimized Weight 
	Bias = sess.run(B) # Optimized Bias 
	
	# Final Accuracy 
	correct_prediction = tf.equal(tf.argmax(Y_hat, 1), 
									tf.argmax(Y, 1)) 
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 
											tf.float32)) 
	print("\nAccuracy:", accuracy_history[-1], "%") 

######################################################
plt.plot(list(range(epochs)), cost_history) 
plt.xlabel('Epochs') 
plt.ylabel('Cost') 
plt.title('Decrease in Cost with Epochs') 

plt.show() 

plt.plot(list(range(epochs)), accuracy_history) 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.title('Increase in Accuracy with Epochs') 

plt.show() 

plt.show() 


# Calculating the Decision Boundary 
decision_boundary_x = np.array([np.min(x_orig[:, 0]), 
							np.max(x_orig[:, 0])]) 

decision_boundary_y = (- 1.0 / Weight[0]) * (decision_boundary_x * Weight + Bias) 

decision_boundary_y = [sum(decision_boundary_y[:, 0]), 
					sum(decision_boundary_y[:, 1])] 

# Positive Data Points 
x_pos = np.array([x_orig[i] for i in range(len(x_orig)) 
									if y_orig[i] == 1]) 

# Negative Data Points 
x_neg = np.array([x_orig[i] for i in range(len(x_orig)) 
									if y_orig[i] == 0]) 

# Plotting the Positive Data Points 
plt.scatter(x_pos[:, 0], x_pos[:, 1], 
color = 'blue', label = 'Positive') 

# Plotting the Negative Data Points 
plt.scatter(x_neg[:, 0], x_neg[:, 1], 
color = 'red', label = 'Negative') 

# Plotting the Decision Boundary 
plt.plot(decision_boundary_x, decision_boundary_y) 
plt.xlabel('Feature 1') 
plt.ylabel('Feature 2') 
plt.title('Plot of Decision Boundary') 
plt.legend() 

plt.show() 



    


