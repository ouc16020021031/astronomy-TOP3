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
drop_clos = ['id', 'sum']
data = pd.read_pickle(path_pickle+'test.pickle')

encoder = LabelEncoder().fit(['galaxy', 'qso', 'star'])
num_classes = len(encoder.classes_ )

test_data_use = data.drop(drop_clos, axis=1)
test_data_use = np.array(test_data_use)
test_data_use = test_data_use.reshape([test_data_use.shape[0], test_data_use.shape[1], 1, 1])

data = data[['id']]

#train
fold_n = 10
pred_proba = np.zeros([test_data_use.shape[0], num_classes])
for i in range(num_classes):
    data['class%s'%i] = 0

for i in range(fold_n):
    print('Fold', str(i+1)+'/'+str(fold_n))
    modelPath = path_model+"fold%s_best.hdf5"%i
    model = resnet.resnet_34(test_data_use.shape[1], test_data_use.shape[2], test_data_use.shape[3], num_classes)
    Optimizer = RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-6)
    model.load_weights(modelPath, by_name=True)
    model.compile(loss='categorical_crossentropy', optimizer=Optimizer, metrics=['acc'])
    model.load_weights(modelPath, by_name=True)
    pred_proba  += model.predict(test_data_use, verbose=1, batch_size=1024)

for i in range(num_classes):
    data.loc[:, 'class%s'%i] = pred_proba[:, i]

data.to_csv(path_result+'pred.csv', index=False)