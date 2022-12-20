#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Convolutional Neural Network with generative network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[ ]:


# Initialising the CNN
classifier = Sequential()


# In[ ]:


# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (224, 224, 3), activation = 'relu'))


# In[ ]:


# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
##classifier.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))


# In[ ]:


# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[ ]:


## Adding a third convolutional layer
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[ ]:


# Step 3 - Flattening
classifier.add(Flatten())


# In[ ]:


# Step 4 - Full connection
#classifier.add(Dense(output_dim = 64, activation = 'relu'))
classifier.add(Dense(units=64, activation='relu'))
#classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
classifier.add(Dense(units=1, activation='sigmoid'))


# In[ ]:


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (224, 224),
                                                 batch_size = 2,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (224, 224),
                                            batch_size = 2,
                                            class_mode = 'binary')

classifier.fit(training_set,
                         steps_per_epoch = 100,
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = 30)


# In[ ]:


#saving the model architecture and weights via hdf5 
from keras.models import load_model
from keras.preprocessing import image


# In[ ]:


classifier.save('model_weights.h5')  # creates a HDF5 file 'my_model.h5'


# In[ ]:


#from keras.models import Sequential
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
#from keras.preprocessing import image


# In[ ]:


# returns a compiled model
# identical to the previous one
classifier1 = load_model('model_weights.h5')

predicted_normal = 0
predicted_infected = 0
actual_normal = 0
actual_infected = 0

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0


# In[ ]:


#Image index i for infected
for i in range(1,8):
	#Predicting the dataset
	image_name = 'infected.'+str(i)+'.jpg'
	file_name = 'dataset/test_set/infected/infected.'+str(i)+'.jpg'
	
	test_image = image.load_img(file_name, target_size = (224, 224))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier1.predict(test_image)
	
	# training_set.class_indices
	if result[0][0] == 1:
		predicted_infected += 1
		true_positive += 1
	    # print('Model prediction for image '+image_name+':  infected')
	else:
		predicted_normal += 1
		false_negative += 1
	    # print('Model prediction for image '+image_name+':   NORMAL')
	actual_infected += 1


# In[ ]:


#Image index i for normal
for i in range(1,8):
	#Predicting the dataset
	image_name = 'normal.'+str(i)+'.jpg'
	file_name = 'dataset/test_set/normal/normal.'+str(i)+'.jpg'
	
	test_image = image.load_img(file_name, target_size = (224, 224))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier1.predict(test_image)
	
	# training_set.class_indices
	if result[0][0] == 1:
		predicted_infected += 1
		false_positive += 1
	    # print('Model prediction for image '+image_name+':  infected')
	else:
		predicted_normal += 1
		true_negative += 1
	    # print('Model prediction for image '+image_name+':   NORMAL')
	actual_normal += 1


# In[ ]:


print("Predicted_Normal: ",predicted_normal)
print("Actual_Normal: ",actual_normal)
print("Predicted_infected: ",predicted_infected)
print("Actual_infected: ",actual_infected)
print()
print("True Positive: ",true_positive)
print("True Negative: ",true_negative)
print("False Positive: ",false_positive)
print("False Negative: ",false_negative)


# In[ ]:


A = true_positive + false_positive


# In[ ]:


Accuracy= (true_positive/A)*100


# In[ ]:


print(Accuracy)

