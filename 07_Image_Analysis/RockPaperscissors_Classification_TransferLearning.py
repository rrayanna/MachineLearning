import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

def Inception():
    # Download the inception v3 weights
    url = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    urllib.request.urlretrieve(url, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # Import the inception model
    from tensorflow.keras.applications.inception_v3 import InceptionV3

    # Create an instance of the inception model from the local pre-trained weights
    local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    pre_trained_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = None)
    pre_trained_model.load_weights(local_weights_file)

    for layer in pre_trained_model.layers:
      layer.trainable = False

    return pre_trained_model

#Get the data
url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
urllib.request.urlretrieve(url, 'rps.zip')
local_zip = 'rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/')
zip_ref.close()

TRAINING_DIR = "tmp/rps/"

#Load the data using ImageDataGenerator
training_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest', validation_split=0.1)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR, batch_size=32, class_mode='categorical', target_size=(150, 150), subset='training')

train_generator = training_datagen.flow_from_directory(TRAINING_DIR, batch_size=32, class_mode='categorical', target_size=(150, 150), subset='training')

validation_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)
validation_generator = validation_datagen.flow_from_directory(TRAINING_DIR, batch_size=32, class_mode='categorical', target_size=(150, 150), subset='validation')

# Build the model
pre_trained_model = Inception()
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = tf.keras.layers.Flatten()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = tf.keras.layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.2
x = tf.keras.layers.Dropout(0.2)(x)
# Add a final softmax layer for classification
x = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(pre_trained_model.input, x)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

# Callbacks
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.950):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True
      
callbacks = myCallback()

# Train the model
history = model.fit_generator(train_generator, steps_per_epoch=63, epochs=10,
validation_data=validation_generator, callbacks = [callbacks]) 
