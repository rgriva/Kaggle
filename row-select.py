# This selects a bigger sample from the full train set

# Opening it as in the timing-data script
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

# Determine lines to be skipped
total = 184903891
frac = 0.9
size = round(frac * total)

# Generating sample index
skip = np.random.choice(np.arange(1, total), size=size, replace=False)
gc.collect()

full_data = pd.read_csv('/home/raul.guarini/mmd/train_reduced.csv',
                        dtype=dtypes, usecols=cols, skiprows=skip)
