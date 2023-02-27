import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from  pathlib import Path
import os
# for dirname, _, filenames in os.walk('../data'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

dataDir = '../data/'
dataPath = Path(dataDir)
# for file in dataPath.rglob('*.csv'):
#     print(file)


train_data = pd.read_csv("../data/train.csv")
train_data.head()