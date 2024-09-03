import os
import shutil

from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0  
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# Set the paths to the original dataset and the new data directories
original_dataset_dir = r'C:\Users\geeth\Downloads\leaf'

base_dir = r'C:\Users\geeth\Downloads\leaf'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# List of class names
class_names = ['bacterial_leaf_blight', 'Brown_spot', 'Healthy', 'Hispa', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'Shath Blight', 'Tungro']

# Create base directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)
################
# Create train, validation, and test directories inside the base directory
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split the data into train, validation, and test sets (adjust the test size and validation size as needed)
for class_name in class_names:
    class_dir = os.path.join(original_dataset_dir, class_name)
    images = os.listdir(class_dir)
    # Split the data into train and remaining data
    train_images, remaining_images = train_test_split(images, test_size=0.2, random_state=42)
    # Further split the remaining data into validation and test sets
    val_images, test_images = train_test_split(remaining_images, test_size=0.5, random_state=42)
    # Create class directories inside train, validation, and test directories
    train_class_dir = os.path.join(train_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)

    val_class_dir = os.path.join(validation_dir, class_name)
    os.makedirs(val_class_dir, exist_ok=True)

    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(test_class_dir, exist_ok=True)
##################
# Move images to their respective directories
for train_image in train_images:
    src_path = os.path.join(class_dir, train_image)
    dest_path = os.path.join(train_class_dir, train_image)
    shutil.copy(src_path, dest_path)

for val_image in val_images:
    src_path = os.path.join(class_dir, val_image)
    dest_path = os.path.join(val_class_dir, val_image)
    shutil.copy(src_path, dest_path)

for test_image in test_images:
    src_path = os.path.join(class_dir, test_image)
    dest_path = os.path.join(test_class_dir, test_image)
    shutil.copy(src_path, dest_path)

batch_size = 16
image_size = (224, 224)
###################
# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)
################
# Load the EfficientNetB0 model with pre-trained weights
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom dense layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(9, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set the number of epochs
epochs = 5

#################
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
)
# Plot the training and validation accuracy and loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(loss) + 1)

plt.plot(epochs_range, loss, 'y', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(epochs_range, accuracy, 'y', label='Training acc')
plt.plot(epochs_range, val_accuracy, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)
print("Test accuracy: ", accuracy)

###################
# Get the true labels and predicted labels for the test set
true_labels = test_generator.classes

predictions = model.predict(test_generator)
predicted_labels = tf.argmax(predictions, axis=1)

# Create the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.show()
