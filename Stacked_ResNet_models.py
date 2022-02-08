# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:06:25 2022

@author: Asma Baccouche
"""
import matplotlib.pyplot as plt
import itertools
import keras
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from keras.optimizers import Adam
import sklearn, glob
from xgboost import XGBClassifier
from keras.preprocessing.image import ImageDataGenerator

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

data = 'cbis_private'
task = ''

if task == '':
    d = data
else:
    d = data + '_' + task

path = 'Classification files/Classification data/multiple datasets/'+d

if task == 'birads':
    if 'cbis' in d:
        nb = 4
    else:
        if 'inbreast' in d or 'all' in d:
            nb = 5
        else:
            if 'private' in d:
                nb = 2
if task == 'shape':
    nb = 4
    
if task == '':
    nb = 2
    
train_path = path+'/train_dir'
valid_path = path+'/val_dir'
test_path = path+'/test_dir'

if 'birads' in d:
    if nb == 2:
        cm_plot_labels = ['B4', 'B5']
        loss_fn = BinaryCrossentropy(label_smoothing=0.25)
    if nb == 4:
        cm_plot_labels = ['B2', 'B3', 'B4', 'B5']
        loss_fn = CategoricalCrossentropy(label_smoothing=0.25)
    if nb == 5:
        cm_plot_labels = ['B2', 'B3', 'B4', 'B5', 'B6']
        loss_fn = CategoricalCrossentropy(label_smoothing=0.25)
    c = 1024
    
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
    
elif 'shape' in d:
    cm_plot_labels = ['IRREGULAR', 'LOBULATED', 'OVAL', 'ROUND']
    loss_fn = CategoricalCrossentropy(label_smoothing=0.25)
    c = 100
    
    num_train_samples = len(glob.glob(train_path+'/IRREGULAR/*.png'))+len(glob.glob(train_path+'/LOBULATED/*.png'))+\
    len(glob.glob(train_path+'/OVAL/*.png'))+len(glob.glob(train_path+'/ROUND/*.png'))
    num_test_samples = len(glob.glob(test_path+'/IRREGULAR/*.png'))+len(glob.glob(test_path+'/LOBULATED/*.png'))+\
    len(glob.glob(test_path+'/OVAL/*.png'))+len(glob.glob(test_path+'/ROUND/*.png'))
    num_val_samples = len(glob.glob(valid_path+'/IRREGULAR/*.png'))+len(glob.glob(valid_path+'/LOBULATED/*.png'))+\
    len(glob.glob(valid_path+'/OVAL/*.png'))+len(glob.glob(valid_path+'/ROUND/*.png'))
    train_batch_size = 32
    val_batch_size = train_batch_size
    test_batch_size = train_batch_size
    image_size = 224
    
else:
    cm_plot_labels = ['Malignant', 'Benign']
    loss_fn = BinaryCrossentropy(label_smoothing=0.25)
    c = 1024

    num_train_samples = len(glob.glob(train_path+'/malig/*.png'))+len(glob.glob(train_path+'/benig/*.png'))
    num_val_samples = len(glob.glob(valid_path+'/malig/*.png'))+len(glob.glob(valid_path+'/benig/*.png'))
    num_test_samples = len(glob.glob(test_path+'/malig/*.png'))+len(glob.glob(test_path+'/benig/*.png'))
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

l = []
ll =[]
for i in range(int(train_steps)):
    x, y = next(train_batches)
    l.append(x)
    ll.append(y)
    
x_train = l[0]
y_train = ll[0]
for i in range(1, len(l)):
    x_train = np.append(x_train, l[i], axis = 0)
    y_train = np.append(y_train, ll[i], axis = 0)
    
l2 = []
ll2 =[]
for i in range(int(test_steps)):
    x, y = next(test_batches)
    l2.append(x)
    ll2.append(y)
    
x_test = l2[0]
y_test = ll2[0]
for i in range(1, len(l2)):
    x_test = np.append(x_test, l2[i], axis = 0)
    y_test = np.append(y_test, ll2[i], axis = 0)    

model1 = keras.applications.resnet_v2.ResNet50V2()
model2 = keras.applications.resnet_v2.ResNet101V2()
model3 = keras.applications.resnet_v2.ResNet152V2()

models = [model1, model2, model3]

for i in range(len(models)):
    x = models[i].output
    x = Dense(c, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(nb, activation='softmax')(x)
    models[i] = Model(inputs=models[i].input, outputs=predictions)  


path = 'Classification files/Models weights/'
#path = 'Classification files/Classification data/single datasets/'

p = path + 'cbis'

model1 = models[0]        
model1.load_weights(p+'_ResNet50V2_model.h5')
model2 = models[1]       
model2.load_weights(p+'_ResNet101V2_model.h5')
model3 = models[2]       
model3.load_weights(p+'_ResNet152V2_model.h5')

p1 = model1.predict(x_test)
p2 = model2.predict(x_test)
p3 = model3.predict(x_test)

d = 'cbis'

np.save('Classification files/Predictions/pred_'+d+'_ResNet50V2.npy', p1)
np.save('Classification files/Predictions/pred_'+d+'_ResNet101V2.npy', p2)
np.save('Classification files/Predictions/pred_'+d+'_ResNet152V2.npy', p3)

model1 = Model(inputs=model1.input, outputs=model1.get_layer('dropout').output)
model2 = Model(inputs=model2.input, outputs=model2.get_layer('dropout_'+str(1)).output)
model3 = Model(inputs=model3.input, outputs=model3.get_layer('dropout_'+str(2)).output)

#y1t = model1.predict_generator(train_batches,steps=train_steps, verbose=0)
#y2t = model2.predict_generator(train_batches,steps=train_steps, verbose=0)
#y3t = model3.predict_generator(train_batches,steps=train_steps, verbose=0)
#
#y1 = model1.predict_generator(test_batches,steps=test_steps, verbose=0)
#y2 = model2.predict_generator(test_batches,steps=test_steps, verbose=0)
#y3 = model3.predict_generator(test_batches,steps=test_steps, verbose=0)

y1t = model1.predict(x_train)
y2t = model2.predict(x_train)
y3t = model3.predict(x_train)

y1 = model1.predict(x_test)
y2 = model2.predict(x_test)
y3 = model3.predict(x_test)

models = [model1, model2, model3]

def define_stacked_model(members):
  
    ensemble_outputs = []
    for i in range(len(members)):
        ipt = keras.layers.Input((c))
        ipt._name = 'ensemble_' + str(i+1) + '_' + ipt.name
        ensemble_outputs.append(ipt)

    merge = concatenate(ensemble_outputs)
    hidden = Dense(1000, activation='sigmoid')(merge)   
    hidden2 = Dense(100, activation='relu')(hidden)
    hidden3 = Dense(10, activation='sigmoid')(hidden2)    
    output = Dense(nb, activation='softmax')(hidden3)
    
    model = Model(inputs=ensemble_outputs, outputs=output)
    model.compile(loss=loss_fn,
                  metrics=['accuracy'],
                  optimizer=Adam(lr=0.000001))
    return model

##0.00001

filepath = p+"_stacked_ResNet_models.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2,
                              verbose=1, mode='max', min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

stacked_model = define_stacked_model(models)

#stacked_model.fit([y1t, y2t, y3t], y_train, epochs=30, verbose=1, callbacks=callbacks_list)
stacked_model.load_weights(filepath)

predictions = stacked_model.predict([y1, y2, y3], verbose=1)

cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))

plot_confusion_matrix(cm, cm_plot_labels)

print('Accuracy: ', sklearn.metrics.accuracy_score(y_test.argmax(axis=1), predictions.argmax(axis=1)))
print(sklearn.metrics.classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=cm_plot_labels))
try:
    print('AUC score: ', sklearn.metrics.roc_auc_score(y_test.argmax(axis=1), predictions, multi_class="ovr", average='macro'))
except:
    print('AUC score: ', sklearn.metrics.roc_auc_score(y_test.argmax(axis=1), predictions.argmax(axis=1)))

np.save('Classification files/Predictions/pred_'+d+'_stacked.npy', predictions)
stacked_model.save_weights(filepath)

# XGBoost
y = np.average([y1, y2, y3], axis=0)
yt = np.average([y1t, y2t, y3t], axis=0)
model = XGBClassifier(max_depth=1)
model.fit(yt, y_train.argmax(axis=1))

filepath2 = p+"_average_ResNet_models.h5"

yhat = model.predict(y)
yhatt = model.predict_proba(y)

cm = confusion_matrix(y_test.argmax(axis=1), yhat)

plot_confusion_matrix(cm, cm_plot_labels)

print('Accuracy: ', sklearn.metrics.accuracy_score(y_test.argmax(axis=1), yhat))
print(sklearn.metrics.classification_report(y_test.argmax(axis=1), yhat, target_names=cm_plot_labels))
try:
    print('AUC score: ', sklearn.metrics.roc_auc_score(y_test.argmax(axis=1), yhatt, multi_class="ovr", average='macro'))
except:
    print('AUC score: ', sklearn.metrics.roc_auc_score(y_test.argmax(axis=1), yhatt.argmax(axis=1)))

np.save('Classification files/Predictions/pred_'+d+'_average.npy', yhat)
model.save_model(filepath2)













