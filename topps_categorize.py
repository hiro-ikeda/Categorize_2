from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras import backend as K

# Image Size
img_width, img_height = 224, 224

#Image Path
train_data_dir = ""
validation_data_dir = ""
result_dir = ""

# Num of Samples
nb_train_samples = 0
nb_validation_samples = 0

epochs = 50
batch_size = 16

nb_classes = 0
classes = ["",""]


"""
input_tensor = Input(shape=(img_width, img_height))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation="relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))

model = Model(input=vgg16.input, output=top_model(vgg16.output))

for layer in model.layers[:15] :
    layer.trainable = False
"""

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)
"""
history = model.fit_generator(
    train_generator,
    samples_per_epoch = nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_validation_samples=nb_validation_samples)

model.save_weights(os.path.join(result_dir, 'finetuning.h5'))
save_history(history, os.path.join(result_dir, 'history_finetuning.txt'))
"""
