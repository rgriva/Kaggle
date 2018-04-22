# This is hardly inspired by the work of Raven Ron on
# https://www.kaggle.com/codeastar/random-forest-classification-on-talkingdata

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# print(os.listdir("../input"))

import gc
import time

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

# Correcting path to data inside EMAp servers
path = '/dados/Dados/Kaggle/'


def handleClickHour(df):
    df['click_hour'] = (pd.to_datetime(df['click_time']).dt.round('H')).dt.hour
    df['click_hour'] = df['click_hour'].astype('uint16')
    df = df.drop(['click_time'], axis=1)
    return df


train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']

# Importing data

# Load training df (partly)
start_time = time.time()
df_train_30m = pd.read_csv(path + 'train.csv', dtype=dtypes,
                           skiprows=range(1, 133333333), nrows=33333333, usecols=train_columns)
print('Load df_train_30m with {} seconds'.format(round(time.time() - start_time, 3)))

# Load testing df
start_time = time.time()
df_test = pd.read_csv(path + 'test.csv', dtype=dtypes)
print('Load df_test with {} seconds'.format(round(time.time() - start_time, 3)))

train_record_index = df_train_30m.shape[0]

# handle click hour
df_train_30m = handleClickHour(df_train_30m)
df_test = handleClickHour(df_test)
gc.collect()
print('ClickTime data correctly handled.')

# df for submit
df_submit = pd.DataFrame()
df_submit['click_id'] = df_test['click_id']

# Extracting learning data
Learning_Y = df_train_30m['is_attributed']
print('Training target correctly extracted.')

# drop zone
df_test = df_test.drop(['click_id'], axis=1)
df_train_30m = df_train_30m.drop(['is_attributed'], axis=1)
gc.collect()

df_merge = pd.concat([df_train_30m, df_test])
del df_train_30m, df_test
gc.collect()
print('Data was correctly merged')

# Count ip for both train and test df
start_time = time.time()
df_ip_count = df_merge['ip'].value_counts().reset_index(name='ip_count')
df_ip_count.columns = ['ip', 'ip_count']
print('Loaded df_ip_count with {} seconds'.format(round(time.time() - start_time, 3)))

print('Starting to merge with main dataset...')
df_merge = df_merge.merge(df_ip_count, on='ip', how='left', sort=False)
df_merge['ip_count'] = df_merge['ip_count'].astype('uint16')
print('Merging operation completed.')

df_merge = df_merge.drop(['ip'], axis=1)
del df_ip_count
gc.collect()

df_train = df_merge[:train_record_index]
df_test = df_merge[train_record_index:]

del df_merge
gc.collect()

# Start of Random Forest Implementation
from sklearn.ensemble import RandomForestClassifier

print('Starting to fit Random Forest Model... The machine is learning...')
start_time = time.time()
rf = RandomForestClassifier(n_estimators=13, max_depth=13, random_state=13, verbose=2, n_jobs=18)
rf.fit(df_train, Learning_Y)
print('The machine heas learned.')
print('RandomForest has fitted df_train_30m with {} seconds'.format(round(time.time() - start_time), 3))

# Predict
print('Starting prediction phase...')
start_time = time.time()
predictions = rf.predict_proba(df_test)
print('Prediction done. Elapsed time: {} seconds'.format(round(time.time() - start_time, 3)))

# Creating the submission dataset
df_submit['is_attributed'] = predictions[:, 1]
df_submit.describe()
print('Submission dataset created.')

# Preparing submssion
df_submit.to_csv('random_forest_talking_data.csv', index=False)
print('Submission dataset saved correctly.')
