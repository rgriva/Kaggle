# This is just a script for creating columns with timestamp data from original
# data. It reproduces the Timing_Data notebook

import pandas as pd
import numpy as np
import gc

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
}

# We delete the attributed_time column since it has a lot of missing values and might not be so informative
# Reference: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/51411
cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']

full_data = pd.read_csv(
    '/Users/Raul/Dropbox/Code/raul-mmd/kaggle/train_sample_reduced.csv', dtype=dtypes, usecols=cols)

# Cleaning the environment
gc.collect()

# This will take a while!!!
v = full_data.click_time.str.split()
full_data['days'] = v.str[0].astype('uint8')
full_data[['hours', 'minutes', 'seconds']] = (
    pd.to_timedelta(v.str[-1]).dt.components.iloc[:, 1:4]
).astype('uint8')

# Deleting 'click_time':
full_data = full_data.drop(['click_time'], 1)
gc.collect()

# Saving the results!
pd.DataFrame.to_csv(full_data, '/Users/Raul/Dropbox/Code/raul-mmd/kaggle/train_sample_timed.csv')
