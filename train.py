import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import random
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
  
import keras
from keras_radam import RAdam
from keras import backend as K
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score, precision_score, recall_score

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from tools import Metrics
import resnet

#setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 0.9 
session = tf.Session(config=config)
KTF.set_session(session)

path = os.getcwd()
path_sub = path + '/../data/sub/'
path_data = path + '/../data/raw/'
path_model = path + '/../data/model/'
path_result = path + '/../data/result/'
path_pickle = path + '/../data/pickle/'
path_profile = path + '/../data/profile/'

for i in [path + '/../data/', path_sub, path_data, path_model, path_result, path_pickle, path_profile]:
    try:
        os.mkdir(i)
    except:
        pass

#ready
label = 'answer'
drop_clos = [label, 'id', 'sum']
data = pd.read_pickle(path_pickle+'data.pickle')

train_label = data[label]
encoder = LabelEncoder()
train_label = encoder.fit_transform(train_label)
num_classes = len(encoder.classes_ )

train_data_use = data.drop(drop_clos, axis=1)
train_data_use = np.array(train_data_use)
train_data_use = train_data_use.reshape([train_data_use.shape[0], train_data_use.shape[1], 1, 1])

data = data[['id', label]]

#train
fold_n = 10
kfold = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=2019)
kf = kfold.split(train_data_use, train_label)
for i in range(num_classes):
    data['class%s'%i] = 0
    
for i, (train_fold, validate) in enumerate(kf):
    print('Fold', str(i+1)+'/'+str(fold_n))
    train_labels = to_categorical(train_label, num_classes=num_classes)
    X_train, X_validate, label_train, label_validate = train_data_use[
        train_fold], train_data_use[validate], train_labels[train_fold], train_labels[validate]
    
    monitor = 'val_acc'
    modelPath = path_model+"fold%s_best.hdf5"%i
    early_stopping = EarlyStopping(monitor=monitor, patience=5, mode='max', verbose=0)
    plateau = ReduceLROnPlateau(factor=0.1, monitor=monitor, patience=2, mode='max', verbose=0)
    checkpoint = ModelCheckpoint(modelPath, save_best_only=True, monitor=monitor, mode='max', verbose=1)
    callbacks_list = [Metrics(valid_data=(X_validate, label_validate)), early_stopping, plateau, checkpoint]
    
    model = resnet.resnet_34(train_data_use.shape[1], train_data_use.shape[2], train_data_use.shape[3], num_classes)
    Optimizer = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-6)
    try:
        model.load_weights(modelPath, by_name=True)
    except:
        model.compile(loss='categorical_crossentropy', optimizer=Optimizer, metrics=['acc'])
        history = model.fit(X_train, label_train, epochs=100, batch_size=128,
                            verbose=0, shuffle=True, callbacks=callbacks_list, 
                            validation_data=(X_validate, label_validate))
        model.load_weights(modelPath, by_name=True)
        
    oof_proba  = model.predict(X_validate, verbose=1, batch_size=1024)
    print('Best score:', f1_score(label_validate.argmax(axis=-1), np.array(oof_proba).argmax(axis=-1), average='macro'))
    for i in range(num_classes):
        data.loc[validate, 'class%s'%i] = oof_proba[:, i]

data.to_csv(path_result+'oof.csv', index=False)
