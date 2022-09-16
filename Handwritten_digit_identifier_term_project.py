#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install --upgrade pip


# In[6]:


get_ipython().system('pip install -q gradio')


# In[10]:


#importing necessary libraries
import tensorflow as tf
import gradio as gr

#loading mnist dataset in training and testing variables
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#rescaling the data before feeding it to the model by dividing it by 255 (pixel value)
x_train = x_train / 255.0, 
x_test = x_test / 255.0


# In[11]:


#using sequential model where we have flattend the image in 28 x 28 pixel
#we have used three layers where in the second hidden layer we have 128 nodes and using relu activation function
#for the third hidden layer we have only 10 nodes and used softmax function
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

#Ccompile” the model with an appropriate loss function, optimizer, and choose which metrics to display during training with 10 epochs
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)


# In[12]:


#function to classify the digit where it will reshape the input given by the user and predict it
def classify(input):
    prediction = model.predict(input.reshape(1, 28, 28)).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

#for the GUI we have used gradio
#sketchpad = gr.inputs.Sketchpad()
label = gr.outputs.Label(num_top_classes=3)
interface = gr.Interface(classify, inputs="sketchpad", outputs=label, live=True, capture_session=True)


# In[13]:


#launching the interface
interface.launch(share=True);


# In[ ]:





# In[ ]:





# In[ ]:




