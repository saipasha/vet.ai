import urllib.request
import zipfile
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download book horse vs human dataset
# Download training imgs
url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/ horse-or-human.zip"
file_name = "horse-or-human.zip"
training_dir = './datasets/horse_vs_human_data/training/'
urllib.request.urlretrieve(url, file_name)
zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

# Download validation data
validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com /validation-horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"
validation_dir = './datasets/horse_vs_human_data/validation/'
urllib.request.urlretrieve(validation_url, validation_file_name)
zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()

# Train data generator
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
  training_dir,
  target_size=(300, 300),
  class_mode='binary'
)

# Validation data generator
validation_datagen = ImageDataGenerator(rescale=1/255)
validation_generator = train_datagen.flow_from_directory(
  validation_dir,
  target_size=(300, 300),
  class_mode='binary'
)


# Example CNN architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the network
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

# Train the network
history = model.fit_generator(train_generator, epochs=15, validation_data=validation_generator)

# Test the model
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
image_tensor = np.vstack([x])
classes = model.predict(image_tensor)
print(f"classes: {classes}")
