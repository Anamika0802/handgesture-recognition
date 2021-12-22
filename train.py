import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,classification_report


data_gen_test = ImageDataGenerator(rescale = 1./255)
#data augmentation
data_gen_train = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range= 0.2,horizontal_flip= True)
#train set
training_data = data_gen_train.flow_from_directory('data/train',target_size = (64,64),batch_size = 10,
                color_mode = 'grayscale',class_mode= 'categorical')

#test data
test_data = data_gen_test.flow_from_directory('data/test',target_size = (64,64),batch_size = 10,
                color_mode = 'grayscale',class_mode= 'categorical')
x_test,y_test = test_data.next()
#model
model = models.Sequential([
    #cnn layers
    layers.Conv2D(filters = 32, activation='relu', kernel_size = (3,3),input_shape = (64,64,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters = 32, activation='relu', kernel_size = (3,3)),
    layers.MaxPooling2D((2,2)),
    #dense
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout((0.30)),
    layers.Dense(96, activation = 'relu'),
    layers.Dropout((0.30)),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(11, activation = 'softmax')
])
model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(training_data,epochs =10, validation_data = test_data,
            steps_per_epoch = 800,#no. of images in training data
            validation_steps = 200)

model_json = model.to_json()
with open("model-bw.json","w") as json_file:
    json_file.write(model_json)
model.save_weights('model-bw.h5')

#model - data2 71% val,99% acc, 64, all threshold
#model - data3 91% val,99.5% acc, 64, no threshold
#model - data3 91% val,99.5% acc, 64, no threshold,dropout
















































































































































































































































































































#matrics
    # predicted_data = model.predict(x_test)
    # y_test_encoded = np.argmax(y_test, axis = 1)
    # pred_data = [np.argmax(element) for element in predicted_data]
    # # print("\ntest encoded\n",y_test_encoded[:5])
    
    # # print("\npred\n",pred_data[:5])
    # print(model.evaluate(test_data))
    # print(confusion_matrix(y_test_encoded,pred_data))
    # mat = classification_report(y_test_encoded,pred_data,zero_division=1)
    # print(mat)
    #saving model