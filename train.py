import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_gen_test = ImageDataGenerator(rescale = 1./255)
#data augmentation
data_gen_train = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range= 0.2,horizontal_flip= True)
#train set
training_data = data_gen_train.flow_from_directory('handgesture-recognition/data/train',target_size = (130,130),batch_size = 10,
                color_mode = 'grayscale',class_mode= 'categorical')

#test data
test_data = data_gen_test.flow_from_directory('handgesture-recognition/data/test',target_size = (130,130),batch_size = 10,
                color_mode = 'grayscale',class_mode= 'categorical')
#model
model = models.Sequential([
    #cnn layers
    layers.Conv2D(filters = 32, activation='relu', kernel_size = (3,3),input_shape = (130,130,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters = 32, activation='relu', kernel_size = (3,3)),
    layers.MaxPooling2D((2,2)),
    #dense
    layers.Flatten(),
    layers.Dense(130, activation = 'relu'),
    layers.Dropout((0.40)),
    layers.Dense(96, activation = 'relu'),
    layers.Dropout((0.40)),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(training_data,epochs =5, validation_data = test_data,
            steps_per_epoch = 200,#no. of images in training data
            validation_steps = 10)