# Part 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#these 2 lines are for timer
from timeit import default_timer as timer
start = timer()
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(activation = 'relu', units = 128))
classifier.add(Dense(activation = 'sigmoid', units = 1))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 63)
# elapsed time in minutes
end = timer()
print("Elapsed time in minutes")
print(0.1*round((end - start)/6))
# end of work message
import os
os.system('say "your program has finished"')

"""
#Predicting New Images
import numpy as np
from skimage.io import imread
from skimage.transform import resize
img = imread('dataset/new/2.jpg') #make sure that path_to_file contains the path to the image you want to predict on. 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)
img = img/(255.0)
prediction = classifier.predict_classes(img)
 
if(prediction):
    print ("DOG")
else:
    print ("CAT")
 """   
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/new/4.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print('dog')
else:
    prediction = 'cat'
    print('cat')

""" # prediction on a new picture
from keras.preprocessing import image as image_utils
import numpy as np
 
test_image = image_utils.load_img('dataset/new/4.jpeg', target_size=(64, 64))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
 
res = classifier.predict_on_batch(test_image)

if(res):
    print ("Dog")
else:
    print ("Cat")"""
