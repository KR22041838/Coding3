#!/usr/bin/env python
# coding: utf-8

# # Week 1 
# # Intro to working with neural nets in keras
# 
# 
# Before running this the first time, please make sure you have (1) installed the necessary Python tools and libraries, and (2) activated your enviromment!
# 
# Please see instructions for this under "Instructions for Setting Up & Using Python & Jupyter" in [moodle](https://moodle.arts.ac.uk/course/view.php?id=71166).

# In[21]:


# issues with how data is written is distributed between x and y train


# In[26]:


import tensorflow as tf
import keras

keras.__version__ #print out current keras version


# In[27]:


tf.__version__ #print out current tensorflow version


# In[28]:


tf.config.experimental.list_physical_devices('GPU') #Do we have access to a GPU?


# In[29]:


# Now do your imports
import numpy as np
from keras.models import Sequential #base keras model
from keras.layers import Dense, Activation #dense = fully connected layer
from tensorflow.keras.optimizers.legacy import SGD #this is just tensorflow.keras.optimizers on earlier versions of tf
import h5py # for saving a trained network

import pandas as pd


# ## Part 1: A very, very simple example
# 
# Can we train a single, 1-input neuron to learn the function output = 5 * input ? 

# In[30]:


#Let's make a training dataset with 3 examples
#x_train = np.array([[2], [1], [-3]]) #Input values for each of our 3 training examples
#y_train = np.array([10, 5, -15]) #Output values (aka "targets") for each of our 3 training examples


# In[52]:


#ChatGTP

#WRITING CVS FILE-------------------------------
#import pandas as pd
#data = pd.DataFrame({'x': [0.107451245188713, 0.608842194080353,0.63094025850296,0.657451987266541,0.602255046367645 ],
#                     'y': [0.726940989494324,0.829100251197815,0.884400963783264,0.820902109146118,0.689584016799927],
#                     'z': [0.0000000611,-0.0000000223,-6.99860009945041E-08,-5.28257011467304E-08,7.72244845848036E-08]})
#data.to_csv('DEBUG.csv', index=False)
#---------------------------------------------------

# Load the training data from a CSV file
train_data = pd.read_csv('DEBUG.csv')

# Extract the input columns from the loaded data
input_columns = ['x', 'y', 'z']
#input_columns = ['x1', 'y1', 'z1','x2','y2','z2','x3','y3','z3','x4','y4','z4','x5','y5','z5','x6','y6','z6','x7','y7','z7','x8','y8','z8','x9','y9','z9','x10','y10','z10','x11','y11','z11','x12','y12','z12','x13','y13','z13','x14','y14','z14','x15','y15','z15','x16','y16','z16','x17','y17','z17','x18','y18','z18','x19','y19','z19','x20','y20','z20','x21','y21','z21','x22','y22','z22','x23','y23','z23','x24','y24','z24','x25','y25','z25','x26','y26','z26','x27','y27','z27','x28','y28','z28','x29','y29','z29','x30','y30','z30','x31','y31','z31','x32','y32','z32','x33','y33','z33','x34','y34','z34','x35','y35','z35','x36','y36','z36','x37','y37','z37','x38','y38','z38','x39','y39','z39','x40','y40','z40','x41','y41','z41','x42','y42','z42']  # Adjust these column names based on your pattern
x_train = train_data[input_columns].values

# Extract the output column from the loaded data
output_column = 'Sign'  # Adjust this column name based on your pattern
y_train = train_data[output_column].values




# In[ ]:





# In[53]:


# Convert the target variable to numeric format, handling scientific notation
#def convert_to_numeric(value):
 #   try:
#        return pd.to_numeric(value)
 #   except (TypeError, ValueError):
 #       return float(value)

#y_train = train_data[output_column].apply(convert_to_numeric).values

# Define a mapping of sign labels to numeric values
sign_mapping = {'A': 1, 'A_1': 2, 'A_2': 3, 'A_3': 4, 'A_4': 5}

# Convert the sign labels in the output column to numeric values
train_data[output_column] = train_data[output_column].map(sign_mapping)

# Extract the input and output data
x_train = train_data[input_columns].values
y_train = train_data[output_column].values


# In[54]:


#---------------------------------------------------
# Create the network using keras
#num_neurons = 1
#model = Sequential() #the basic keras model class
#model.add(Dense(num_neurons, input_dim = 1)) #add a "layer" of 1 neuron, which has 1 input value. This will compute a weighted sum, with no additional activation function.
#model.summary() #print out info about this network


# In[55]:


#Set some training parameters
#use stochastic gradient descent (pretty standard optimisation method) with a learning rate of 0.1 (why not?)
#sgd = SGD(learning_rate=0.1) #Note that in previous versions the "learning_rate" parameter was titled "lr" instead

#optimise the mean squared error (i.e., the mean of the squared difference between the model's output and the training target, for each training example)
#    This is a fairly standard choice when predicting a real value (as opposed to a binary class of 0 or 1)
#model.compile(loss='mean_squared_error', optimizer=sgd)


# In[56]:


# Run the current model on each of the training examples (i.e., using the inputs to calculate the neuron output)
#model.predict(x_train) #outputs garbage as it's not trained yet: It's computing output = w1 * input + w0 using random values of w0 and w1
#--------------------------------------------------------


# In[57]:


# Create the network using keras
#num_neurons = 1
#input_dim = x_train.shape[1]  # Get the number of input features

#model = Sequential()  # The basic keras model class
#model.add(Dense(num_neurons, input_dim=input_dim))  # Add a "layer" of 1 neuron with the correct input dimension
#model.summary()  # Print out info about this network
#--------------------------------------------------------------------
# Create the network using keras update GTP
#num_neurons = 1
#model = Sequential()
#model.add(Dense(num_neurons, input_dim=1))
#model.summary()
#------------------------------------------------------------------------
# Reshape the input data to have shape (None, 1)
x_train = np.reshape(x_train, (-1, 1))

# Define the model architecture
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(1,)))
model.add(Dense(1, activation='linear'))

#---------------------------------------------
# Reshape x_train
x_train = x_train.reshape((3, 5))

# Repeat and concatenate the y values
y_train = np.concatenate([
    np.repeat('A', 3),
    np.repeat('A_1', 3),
    np.repeat('A_2', 3),
    np.repeat('A_3', 3),
    np.repeat('A_4', 3)
], axis=0)


# In[58]:


print(x_train.shape)  # Shape of x_train array
print(y_train.shape)  # Shape of y_train array


# In[59]:


x_train = x_train.reshape(x_train.shape[1],x_train.shape[0])
x_train.shape   # Output: (20, 3)
# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')
# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)


# In[ ]:





# In[60]:


# Run the current model on each of the training examples
model.predict(x_train)  # Outputs initial predictions using random weight values


# In[22]:


#train the model!
model.fit(x_train, y_train, epochs=10)


# In[10]:


#Now run it:
model.predict(x_train) #Does it produce values similar to y_train? It should...


# In[11]:


#we can (and should) run it on some new data, too... 
new_values = np.array([[-10], [0], [5]]) # 3 new "data points"

#Does it output 5 * x for each new value x ?
model.predict(new_values) #be careful to read this using scientific notation ;)


# In[12]:


#Let's look at the model's weights! We'll see (1) the weight for the input, and (2) the weight for the bias
model.get_weights()


# In[13]:


# Perhaps try with a new dataset in which our training example output (target) values are np.array([11, 6, -14])?
y_train_new = np.array([11, 6, -14])
model = Sequential() #the basic model class
model.add(Dense(num_neurons, input_dim = 1)) #add a "layer" of 1 neuron, which has 1 input value. This will compute a weighted sum, with no additional activation function.
model.summary()
sgd = SGD(lr=0.1) 
model.compile(loss='mean_squared_error', optimizer=sgd)
model.fit(x_train, y_train_new, epochs=15)


# In[14]:


model.get_weights() #Notice weight is 5 and bias is now 1 (approximately), encoding the function output = input * 5 + 1


# In[ ]:





# In[ ]:


# Load the test data from a CSV file
test_data = pd.read_csv('test_data.csv')

# Extract the input columns from the loaded data
x_test = test_data[input_columns].values

# Extract the output column from the loaded data
y_test = test_data[output_column].values

