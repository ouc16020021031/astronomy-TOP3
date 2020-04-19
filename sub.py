import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

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
        
#cv_score
oof = pd.read_csv(path_result+'oof.csv')

label ='answer'
encoder = LabelEncoder()
oof[label] = encoder.fit_transform(oof[label])
score = pd.DataFrame(['galaxy', 'qso', 'star', 'all'])
score.columns = ['class']
score['f1'] = 0
score['accuray'] = 0

for i in range(3):
    pred_ = pd.get_dummies(np.array(oof[['class0', 'class1', 'class2']]).argmax(axis=-1))[i]
    label_ = pd.get_dummies(to_categorical(oof[label], num_classes=3).argmax(axis=-1))[i]
    class_ = encoder.inverse_transform([i])[0]
    score.loc[score['class']==class_, 'accuray'] = accuracy_score(label_, pred_)
    score.loc[score['class']==class_, 'f1'] = f1_score(label_, pred_)
    score.loc[score['class']==class_, 'len'] = len(oof[oof[label]==i])
pred_ = pd.get_dummies(np.array(oof[['class0', 'class1', 'class2']]).argmax(axis=-1))
label_ = pd.get_dummies(to_categorical(oof[label], num_classes=3).argmax(axis=-1))

score.loc[score['class']=='all', 'accuray'] = accuracy_score(label_, pred_)
score.loc[score['class']=='all', 'f1'] = f1_score(label_, pred_, average='macro')
score.loc[score['class']=='all', 'len'] = len(oof)

for i in range(4):
    score.loc[i, 'percent'] = score.loc[i, 'len'] / len(oof)
print(score)

#sub_save
pred = pd.read_csv(path_result+'pred.csv')

pred[label] = np.array(pred[['class0', 'class1', 'class2']]).argmax(axis=-1)
pred[label] = encoder.inverse_transform(pred[label])
pred = pred[['id', label]]
pred.columns = ['id', 'label']
pred.to_csv(path_sub+'sub.csv', index=False)