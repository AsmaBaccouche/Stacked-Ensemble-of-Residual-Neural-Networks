# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:28:31 2021

@author: Asma Baccouche
"""

import numpy as np
import keras, glob
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.losses import BinaryCrossentropy
#from keras import backend as K
from keras.layers.core import Dense, Dropout
#from keras.layers import  Conv2D, BatchNormalization, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import sklearn

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



#d = 'cbis'
#path = 'C:/Users/User/OneDrive - University of Louisville/Desktop/'+d
#path = 'Classification files/Classification data/single datasets/'+d

path = 'Classification files/Classification data/multiple datasets/cbis_private'

nb = 2
train_path = path+'/train_dir'
valid_path = path+'/val_dir'
test_path = path+'/test_dir'

num_train_samples = len(glob.glob(train_path+'/malig/*.png'))+len(glob.glob(train_path+'/benig/*.png'))
num_val_samples = len(glob.glob(valid_path+'/malig/*.png'))+len(glob.glob(valid_path+'/benig/*.png'))
num_test_samples = len(glob.glob(test_path+'/malig/*.png'))+len(glob.glob(test_path+'/benig/*.png'))

train_batch_size = 32
val_batch_size = train_batch_size
image_size = 224

#path2 = 'inbreast_patch'
#test_path2 = path2+'/test_dir'
#
#num_test_samples2 = len(glob.glob(test_path2+'/malig/*.png'))+len(glob.glob(test_path2+'/benig/*.png'))

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
test_steps = np.ceil(num_test_samples / train_batch_size)

train_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.resnet_v2.preprocess_input).flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size,
    shuffle=False)

valid_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.resnet_v2.preprocess_input).flow_from_directory(
    valid_path,
    target_size=(image_size, image_size),
    batch_size=val_batch_size,
    shuffle=False)

test_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.resnet_v2.preprocess_input).flow_from_directory(
    test_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size,
    shuffle=False)

mobile = keras.applications.resnet_v2.ResNet50V2()

x = mobile.output
mobile.trainable = False
for layer in mobile.layers:
    if "BatchNormalization" in layer.__class__.__name__:
        layer.trainable = True
    #if "Dense" in layer.__class__.__name__:
    #    layer.trainable = True

x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=predictions)
#model.summary()

model.compile(Adam(lr=0.01), loss=BinaryCrossentropy(label_smoothing=0.25), weighted_metrics=['accuracy'])

#p = 'Classification files/Models weights/'+d
p = 'C:/Users/User/OneDrive - University of Louisville/Desktop/cbis'

p = 'Classification files/Models weights/cbis_inbreast_private'
p = 'Classification files/Models weights/cbis'

filepath = p+"_ResNet50V2_model.h5"


checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

nb_epoch = 30

#model.load_weights("inbreast_ResNet50_v2_model.h5")

history = model.fit_generator(train_batches,
                              steps_per_epoch=train_steps,
                              validation_data=valid_batches,
                              validation_steps=val_steps,
                              epochs=nb_epoch,
                              verbose=1,
                              callbacks=callbacks_list)

model.load_weights(filepath)

val_loss, val_acc= model.evaluate_generator(test_batches, steps=test_steps)

print('val_loss:', val_loss)
print('val_acc:', val_acc)

test_labels = test_batches.classes

predictions = model.predict_generator(test_batches, steps=test_steps, verbose=1)

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

cm_plot_labels = ['Malignant', 'Benign']

plot_confusion_matrix(cm, cm_plot_labels)

print(sklearn.metrics.classification_report(test_labels, predictions.argmax(axis=1), target_names=cm_plot_labels))
print(sklearn.metrics.roc_auc_score(test_labels, predictions.argmax(axis=1)))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(nb_epoch)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
