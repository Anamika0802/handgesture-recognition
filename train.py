import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale = 1./255)

#train set
training_data = data_gen.flow_from_directory('data/train',target_size = (200,200),batch_size = 10,
                color_mode = 'grayscale',class_mode= 'categorical')

#test data
test_data = data_gen.flow_from_directory('data/test',target_size = (200,200),batch_size = 10,
                color_mode = 'grayscale',class_mode= 'categorical')

#model
model = models.Sequential([
    #cnn layers
    layers.Conv2D(filters = 32, activation='relu', kernel_size = (3,3),input_shape = (200,200,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters = 40, activation='relu', kernel_size = (3,3)),
    layers.MaxPooling2D((2,2)),
    #dense
    layers.Flatten(),
    layers.Dense(100, activation = 'relu'),
    layers.Dropout((0.40)),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout((0.40)),
    layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_entropy', matrics = ['accuracy'])
model.fit(training_data,epochs =10, validation_data = test_data,
            steps_per_epoch = 250,#no. of images in training data
            validation_steps = 100)