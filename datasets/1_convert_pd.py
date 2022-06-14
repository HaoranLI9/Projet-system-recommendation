import pickle
import pandas as pd
from utils.util import *

def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df

pre_path = "./datasets/resources" if local_mode else "/tmp"

reviews_df = to_df('%s/%s_5.json' % (pre_path, dataset_name))
with open('%s/reviews_%s.pkl' % (pre_path, dataset_name), 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('%s/meta_%s.json' % (pre_path, dataset_name))
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)
with open('%s/meta_%s.pkl' % (pre_path, dataset_name), 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)
