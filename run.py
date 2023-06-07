import os
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image

# Define constants
img_width, img_height = 64, 64
num_classes = 5
train_epochs = 10
retrain_epochs = 20
batch_size = 64
input_shape = (img_width, img_height, 3)


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder_name, batch_size=batch_size):
        self.annotations = pd.read_csv(
            os.path.join(folder_name, '_annotations.csv'))
        self.folder_name = folder_name
        self.batch_size = batch_size
        self.classes = ['QB', 'DB', 'SKILL', 'LB', 'C']  # list of all classes

    def __len__(self):
        return int(np.ceil(len(self.annotations) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_annotations = self.annotations[idx *
                                             self.batch_size:(idx + 1) * self.batch_size]
        data = []
        labels = []

        for _, row in batch_annotations.iterrows():
            img_path = os.path.join(self.folder_name, row['filename'])
            img = cv2.imread(img_path)
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            cropped_img = cv2.resize(
                img[ymin:ymax, xmin:xmax], (img_width, img_height))
            data.append(cropped_img)
            labels.append(self.classes.index(row['class']))

        data = np.array(data, dtype="float") / 255.0
        labels = tf.keras.utils.to_categorical(
            labels, num_classes=len(self.classes))
        return data, labels


# Create data generators
train_generator = DataGenerator('train')
val_generator = DataGenerator('valid')
test_generator = DataGenerator('test')

# Load the VGG16 network, ensuring the head FC layer sets are left off
base_model = VGG16(weights='imagenet', include_top=False,
                   input_tensor=tf.keras.layers.Input(shape=input_shape))

# Construct the head of the model that will be placed on top of the base model
head_model = base_model.output
head_model = Flatten()(head_model)
head_model = Dense(512, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(num_classes, activation='softmax')(head_model)

# Place the head FC model on top of the base model
model = Model(inputs=base_model.input, outputs=head_model)

# Freeze all the layers in the base model so they will *not* be updated during the training process
for layer in base_model.layers:
    layer.trainable = False

# Compile our model
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# Callbacks for early stopping and best model saving
callbacks = [EarlyStopping(patience=5, restore_best_weights=True),
             ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5')]

# Train the head of the network for a few epochs (all other layers are frozen)
model.fit(train_generator, epochs=train_epochs, validation_data=val_generator, callbacks=callbacks)

# Now that the head FC layers have been trained/initialized, let's
# unfreeze the final set of CONV layers and make them trainable
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile the model with a small learning rate
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics=['accuracy'])

# Train the model again
model.fit(train_generator, epochs=retrain_epochs, validation_data=val_generator, callbacks=callbacks)

# Evaluate the model
score = model.evaluate(test_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model in the new format
model.save('final_model', save_format='tf')

# Define position colors
# 'QB', 'DB', 'SKILL', 'LB', 'C'
colors = {
    'QB': (0, 0, 255),    # Red
    'DB': (255, 0, 0),  # Blue
    'LB': (0, 255, 0),  # Green
    'SKILL' : (230,230,250), # Purple
    'QB' : (255, 165, 0) # Orange
}

image = Image.open('test1.jpeg') # empty image file here

processed_image = image.resize((64,64))

newImg = model.predict(processed_image)

for i in range(64):
    for j in range(64):
        currPos = np.argmax(newImg[i,j])
        color = colors[currPos]
        image[i,j] = color

cv2.imshow(image)