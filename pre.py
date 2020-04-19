import os
import gc
import time
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

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

def pre_chunk(name, l):
    for i in range(2600//l):
        path_temp = path_pickle+'%s_%s.pickle' % (name, i)
        if os.path.exists(path_temp):
            pass
        else:
            data = pd.DataFrame()
            data.to_pickle(path_pickle+'%s_%s.pickle' % (name, i))
            data = pd.read_csv(path_data+'%s.csv' %
                               name, usecols=range(i*l, (i+1)*l))
            start = i*l
            end = (i+1)*l-1
            data = data[(data['FE%s' % start] != 'FE%s' % start)]
            data = data[(data['FE%s' % end] != 'FE%s' % end)]
            data = data.astype(np.float32)
            data.to_pickle(path_pickle+'%s_%s.pickle' % (name, i))
            print(i)

def task_thread(counter, name, l):
    print(f'线程名称：{threading.current_thread().name} 参数：{counter} 开始时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    num = counter
    while num:
        time.sleep(1)
        num -= 1
    pre_chunk(name, l)
    print(f'线程名称：{threading.current_thread().name} 参数：{counter} 结束时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')

def pre(name, l = 200, n_threads=8):
    if os.path.exists(path_pickle+'%s.pickle'%name):
          print(name,'already exist')
          return
    t = []
    print(f'主线程开始时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    for i in range(n_threads):
        t.append(threading.Thread(target=task_thread, args=(i, name, l, )))
    for i in range(n_threads):
        t[i].start()
    for i in range(n_threads):
        t[i].join()
    print(f'主线程结束时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')

    try:
          data = pd.read_csv(path_data+'%s.csv'%name, usecols=[2600,2601])
    except:
          data = pd.read_csv(path_data+'%s.csv'%name, usecols=[2600])
          
    data=data[data['id']!='id']
    for j in range(2600//l):
        path_temp = path_pickle+'%s_%s.pickle'%(name, j)
        temp = pd.read_pickle(path_temp)
        data = pd.concat([data, temp], axis=1)
        os.remove(path_temp)
    data = data[-data['FE0'].isna()]
    data = data.drop_duplicates('id').reset_index().drop('index', axis=1)
    data.to_pickle(path_pickle+'%s.pickle'%name)

# if os.path.exists(path_data+'test_sets.csv'):
#     print('test_sets already exist')
# else:
#     test = pd.DataFrame()
#     for i in tqdm(range(10)):
#         temp = pd.read_csv(path_data+'test_sets_%s.csv' % i)
#         test = pd.concat([test, temp], axis=0)
    
#     test = test.reset_index().drop('index', axis=1)
#     test.to_csv(path_data+'test_sets.csv', index=False)

pre('update_new_columns_trains_sets')
pre('val_sets_v1')
pre('test_sets')

if os.path.exists(path_pickle+'train.pickle'):
    print('train already exist')
else:
    train = pd.read_pickle(path_pickle+'update_new_columns_trains_sets.pickle')
    
    train['sum'] = np.float32(0)
    for i in tqdm(range(2600)):
        train['sum'] += np.float32(train['FE%s' % i]**2)
    train['sum'] = np.sqrt(train['sum']).astype(np.float32)
    for i in tqdm(range(2600)):
        train['FE%s' % i] = np.float32(train['FE%s' % i] / train['sum'])
    
    train.to_pickle(path_pickle+'train.pickle')
    del train
    gc.collect()
    print('train done')

if os.path.exists(path_pickle+'val.pickle'):
    print('val already exist')
else:
    val_label = pd.read_csv(path_data+'val_labels_v1.csv')
    val = pd.read_pickle(path_pickle+'val_sets_v1.pickle')
    val_label.columns = ['id', 'answer']
    val = pd.merge(val_label, val, on='id')
    del val_label
    gc.collect()

    val['sum'] = np.float32(0)
    for i in tqdm(range(2600)):
        val['sum'] += np.float32(val['FE%s' % i]**2)
    val['sum'] = np.sqrt(val['sum']).astype(np.float32)
    for i in tqdm(range(2600)):
        val['FE%s' % i] = np.float32(val['FE%s' % i] / val['sum'])

    val.to_pickle(path_pickle+'val.pickle')
    del val
    gc.collect()
    print('val done')
    
if os.path.exists(path_pickle+'test.pickle'):
    print('test already exist')
else:
    test = pd.read_pickle(path_pickle+'test_sets.pickle')
    
    test['sum'] = np.float32(0)
    for i in tqdm(range(2600)):
        test['sum'] += np.float32(test['FE%s' % i]**2)
    test['sum'] = np.sqrt(test['sum']).astype(np.float32)
    for i in tqdm(range(2600)):
        test['FE%s' % i] = np.float32(test['FE%s' % i] / test['sum'])

    test.to_pickle(path_pickle+'test.pickle')
    del test
    gc.collect()
    print('test done')

if os.path.exists(path_pickle+'data.pickle'):
    print('data already exist')
else:
    train = pd.read_pickle(path_pickle+'train.pickle')
    val = pd.read_pickle(path_pickle+'val.pickle')
    data = pd.concat([train, val], sort=False)
    data = data.reset_index().drop('index', axis=1)
    del val
    gc.collect()
    del train
    data.to_pickle(path_pickle+'data.pickle')
    print('data done')