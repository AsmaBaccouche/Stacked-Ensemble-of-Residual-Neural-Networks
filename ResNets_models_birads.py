# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:36:10 2021

@author: Asma Baccouche
"""

import numpy as np
import keras, glob
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import sklearn
#K.tensorflow_backend._get_available_gpus()

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
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
    plt.show()

path = 'Classification files/Classification data/single datasets/cbis_birads'
nb = 4
train_path = path+'/train_dir'
valid_path = path+'/val_dir'
test_path = path+'/test_dir'

num_train_samples = len(glob.glob(train_path+'/B2/*.png'))+len(glob.glob(train_path+'/B3/*.png'))+\
len(glob.glob(train_path+'/B4/*.png'))+len(glob.glob(train_path+'/B5/*.png'))+len(glob.glob(train_path+'/B6/*.png'))
num_test_samples = len(glob.glob(test_path+'/B2/*.png'))+len(glob.glob(test_path+'/B3/*.png'))+\
len(glob.glob(test_path+'/B4/*.png'))+len(glob.glob(test_path+'/B5/*.png'))+len(glob.glob(test_path+'/B6/*.png'))
num_val_samples = len(glob.glob(valid_path+'/B2/*.png'))+len(glob.glob(valid_path+'/B3/*.png'))+\
len(glob.glob(valid_path+'/B4/*.png'))+len(glob.glob(valid_path+'/B5/*.png'))+len(glob.glob(valid_path+'/B6/*.png'))
train_batch_size = 32
val_batch_size = train_batch_size
test_batch_size = train_batch_size
image_size = 224

train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)
test_steps = np.ceil(num_test_samples / test_batch_size)

train_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.resnet_v2.preprocess_input).flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=train_batch_size)

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
    batch_size=test_batch_size,
    shuffle=False)


model1 = keras.applications.resnet_v2.ResNet50V2()
model2 = keras.applications.resnet_v2.ResNet101V2()
model3 = keras.applications.resnet_v2.ResNet152V2()

models = [model1, model2, model3]

filepath1 = path+"_ResNet50V2_model.h5"
filepath2 = path+"_ResNet101V2_model.h5"
filepath3 = path+"_ResNet152V2_model.h5"

filepaths = [filepath1, filepath2, filepath3]

for i in range(len(models)):
    mobile = models[i]
    filepath = filepaths[i]

    x = mobile.output
    mobile.trainable = False
    for layer in mobile.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True
        if "conv5" in layer.name:
            layer.trainable = True
    
    #x = Dropout(0.3)(x)            
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(nb, activation='softmax')(x)
    
    model = Model(inputs=mobile.input, outputs=predictions)
    #model.summary()
        
    model.compile(Adam(lr=0.01), loss=CategoricalCrossentropy(label_smoothing=0.25), weighted_metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2,
                                  verbose=1, mode='max', min_lr=0.00001)
    
    early_stp = EarlyStopping(monitor='loss', patience=5, mode='min', min_delta=0.1)
    
    callbacks_list = [checkpoint, reduce_lr]
    
    nb_epoch = 30
    
    
    #history = model.fit_generator(train_batches,
    #                              steps_per_epoch=train_steps,
    #                              validation_data=valid_batches,
    #                              validation_steps=val_steps,
    #                              epochs=nb_epoch,
    #                              verbose=1,
    #                              callbacks=callbacks_list)
    
    model.load_weights(filepath)
    
    val_loss, val_acc= model.evaluate_generator(test_batches, steps=test_steps)
    
    test_labels = test_batches.classes
    
    predictions = model.predict_generator(test_batches, steps=test_steps, verbose=0)
    
    cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
    
    if nb == 2:
        cm_plot_labels = ['B4', 'B5']
    elif nb == 4:
        cm_plot_labels = ['B2', 'B3', 'B4', 'B5']
    else:
        cm_plot_labels = ['B2', 'B3', 'B4', 'B5', 'B6']
    
    plot_confusion_matrix(cm, cm_plot_labels)
    
    print('Accuracy: ', val_acc)
    print(sklearn.metrics.classification_report(test_labels, predictions.argmax(axis=1), target_names=cm_plot_labels))
    print('AUC score: ', sklearn.metrics.roc_auc_score(test_labels, predictions, multi_class="ovr", average='macro'))
    #print(sklearn.metrics.roc_auc_score(test_labels, predictions.argmax(axis=1)))


#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs_range = range(nb_epoch)
#
#plt.figure(figsize=(10, 10))
#plt.subplot(2, 2, 1)
#plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(2, 2, 2)
#plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show()

