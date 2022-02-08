# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 10:16:57 2021

@author: Asma Baccouche
"""

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import keras, glob
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

path = 'Classification files/Classification data/single datasets/'
path = 'Classification files/Classification data/multiple datasets/'
path2 = 'Classification files/Predictions/'

d = 'cbis'
task = ''

if task == '':
    p = d
else:
    p = d + '_' + task
    
    
if task == 'shape':
    nb = 4
if task == '':
    nb = 2
if task == 'birads':
    if d == 'cbis':
        nb = 4
    if d == 'inbreast' or d == 'all':
        nb = 5
    if d == 'private':
        nb = 2
    
names = ['ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'average', 'stacked']
models_name = ['ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'Ensemble averaging ', 'Stacked Ensemble']

test_path = path+'cbis_private'+'/test_dir'

if 'birads' in p:
    num_test_samples = len(glob.glob(test_path+'/B2/*.png'))+len(glob.glob(test_path+'/B3/*.png'))+\
    len(glob.glob(test_path+'/B4/*.png'))+len(glob.glob(test_path+'/B5/*.png'))+len(glob.glob(test_path+'/B6/*.png')) 
elif 'shape' in d:
    num_test_samples = len(glob.glob(test_path+'/IRREGULAR/*.png'))+len(glob.glob(test_path+'/LOBULATED/*.png'))+\
    len(glob.glob(test_path+'/OVAL/*.png'))+len(glob.glob(test_path+'/ROUND/*.png'))    
else:
    num_test_samples = len(glob.glob(test_path+'/malig/*.png'))+len(glob.glob(test_path+'/benig/*.png'))

test_steps = np.ceil(num_test_samples / 32)

test_batches = ImageDataGenerator(
    preprocessing_function= \
        keras.applications.resnet_v2.preprocess_input).flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    shuffle=False)

test_labels = test_batches.classes
y_test = to_categorical(test_labels)
 
# macro
fprs=[]
tprs=[]
aucs=[]
for name in names:
    predictions = np.load(path2+'pred_'+p+'_'+name+'.npy')
    if 'average' in name:
        predictions = to_categorical(predictions)
    fpr = dict()
    tpr = dict()
    for i in range(nb):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
       
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i]) 
    mean_tpr /= nb
 
    fprs.append(all_fpr)
    tprs.append(mean_tpr)
    aucs.append(auc(all_fpr, mean_tpr))


if d == 'cbis':
    n = 'CBIS-DDSM'
if d == 'inbreast':
    n = 'INbreast'
if d == 'private':
    n = 'Private'
if d == 'all':
    n = 'Private'
if d == 'cbis_inbreast_private':
    n = 'Private'

if task == 'shape':
    t = 'Shape classification'
if task == 'birads':
    t = 'BI-RADS category classification'
if task == '':
    t = 'Pathology classification'

fig, ax = plt.subplots(1, 1)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_title('ROC curves - ' + t + ' for ' + n)
    
for i in range(len(fprs)):
    ax.plot(fprs[i], tprs[i], label=models_name[i]+' (area = %0.2f)' % aucs[i])
ax.legend(loc="best")
    

# for each class
#fprs=[]
#tprs=[]
#aucs=[]
#for name in names:
#    predictions = np.load('pred_'+p+'_'+name+'.npy')
#    
#    if 'average' in name:
#        predictions = to_categorical(predictions)
#    
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
#    for i in range(nb):
#        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], predictions[:, i])
#        roc_auc[i] = auc(fpr[i], tpr[i])
#    fprs.append(fpr)
#    tprs.append(tpr)
#    aucs.append(roc_auc)
#
#for j in range(nb):
#    fig, ax = plt.subplots(1, 1)
#    ax.plot([0, 1], [0, 1], 'k--')
#    ax.set_xlim([0.0, 1.0])
#    ax.set_ylim([0.0, 1.05])
#    ax.set_xlabel('False Positive Rate (FPR)')
#    ax.set_ylabel('True Positive Rate (TPR)')
#    ax.set_title('ROC curves - CBIS-DDSM class = '+str(j+1))
#    
#    for i in range(len(fprs)):
#        ax.plot(fprs[i][j], tprs[i][j], label=models_name[i]+' (area = %0.2f)' % aucs[i][j])
#    ax.legend(loc="best")
    
