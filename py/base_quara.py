# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
import sys
import re
import gc
import glob

import os
HOME = os.path.expanduser('~')
sys.path.append(f"{HOME}/kaggle/data_analysis/library/")
import utils
from utils import logger_func, get_categorical_features, get_numeric_features, pararell_process
logger = logger_func()
pd.set_option('max_columns', 200)
pd.set_option('max_rows', 200)

train = utils.read_df_pkl('../input/train0*.p')

print(train)
sys.exit()
