

from utils.util import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm
import random
import pickle
import numpy as np
import pandas as pd
import os
import sys
# add PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

context_feautres = {}
behaviour_features = {}
candidate_features = {}
userProfile_features = {}
item2idx_dict = {}
user2idx_dict = {}


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def preprocessing_feature_unixReviewTime(df: pd.DataFrame, numBuckets=1000):
    '''
        discretization of unixReviewTimes
    '''
    unixReviewTime = df['unixReviewTime']
    min_unixReviewTime = unixReviewTime.min()
    unixReviewTime = unixReviewTime - \
        (min_unixReviewTime-1)   # based on min_unixReviewTime
    min_exp, max_exp = 0, int(np.log2(unixReviewTime.max()))
    unixReviewTime_buckets = list(
        map(lambda x: 2**x, range(min_exp, max_exp+2)))
    unixReviewTime_buckets = [0] + unixReviewTime_buckets
    # print(unixReviewTime_buckets) [0, 1, 2, 4, 8, 16, 32, 64, ...]
    df['unixReviewTime'] = pd.cut(
        unixReviewTime, bins=unixReviewTime_buckets, labels=range(1, len(unixReviewTime_buckets)))  # add 1, leave 0 to PAD
    # context_feautres[conf['emb']['context_features']['unixReviewTime_embname']] = nn.Embedding(
    #     len(unixReviewTime_buckets) + 1, conf['emb']['emb_dim'])
    return df, unixReviewTime_buckets


def preprocessing_feature_item_position(df: pd.DataFrame):
    df['item_pos'] = df.sort_values('unixReviewTime').groupby(
        'reviewerID').cumcount() + 1  # start from 1, leave 0 to PAD
    # context_feautres[conf['emb']['context_features']['itempos_embname']] = nn.Embedding(
    #     df['item_pos'].max() + 1, conf['emb']['emb_dim'])
    return df


class _Dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def __len__(self):
        return self.df.size

    def __getitem__(self, idx):
        return self.df.iloc[idx]


def pad_collate_fn(batch):
    '''
        batch: (batch_size, 
    '''
    seq_cols = conf_din['feature_groups']['behaviour_group'] + \
        conf_din['feature_groups']['context_group']
    res = []
    for col in seq_cols:
        res.append(pad_sequence([torch.LongTensor(b[col]) for b in batch]))
    # for b in batch:
    #     print(b)
    #     break
    cons1, cons2, cons3 = [], [], []
    for b in batch:
        cons1.append(b[-3])
        cons2.append(b[-2])
        cons3.append(b[-1])
    res.append(torch.LongTensor(cons1))
    res.append(torch.LongTensor(cons2))
    res.append(torch.LongTensor(cons3))
    return res


def create_amazon_electronic_dataset():
    with open('.. /reviews.pkl', 'rb') as f:
        df = pickle.load(f)
        df = df[['reviewerID', 'asin', 'unixReviewTime']]
    with open('.. /meta.pkl', 'rb') as f:
        meta_df = pickle.load(f)
        meta_df = meta_df[['asin', 'categories']]
        meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

    print('==========Data Preprocess Start============')
    meta_df['categories'], _ = pd.factorize(meta_df['categories'])
    df, unixReviewTime_buckets = preprocessing_feature_unixReviewTime(df)
    df = preprocessing_feature_item_position(df)
    df = meta_df.merge(df, on=['asin', 'asin'])
    df['asin'], item2idx_dict = pd.factorize(df['asin'])
    item2idx_dict = list(item2idx_dict)
    df['reviewerID'], user2idx_dict = pd.factorize(df['reviewerID'])
    user2idx_dict = list(user2idx_dict)
    df['asin'] = df['asin'] + 1
    df['reviewerID'] = df['reviewerID'] + 1  # leave 0 to PAD
    df.columns = ['itemId', 'itemCat', 'userId', 'unixReviewTime', 'itemPos']
    # print(df)
    train_data, val_data, test_data = [], [], []
    seq_len_max = 0
    for user_id, hist in tqdm(df.groupby('userId')):
        pos_list = hist['itemId'].tolist()  # positive sample
        itemCat_list = hist['itemCat'].tolist()
        seq_len_max = max(seq_len_max, len(itemCat_list))

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, len(item2idx_dict) - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))
                    ]  # generate negative sample

        context_unixReviewTime_list = hist['unixReviewTime'].tolist()
        context_itemPos_list = hist['itemPos'].tolist()
        context_group = [[], []]
        behaviour_group = [[], []]
        for i in range(1, len(pos_list)):
            behaviour_group[0].append(pos_list[i - 1])
            behaviour_group[1].append(itemCat_list[i - 1])
            behaviour_group_i = behaviour_group.copy()

            context_group[0].append(context_unixReviewTime_list[i - 1])
            context_group[1].append(context_itemPos_list[i - 1])
            context_group_i = context_group.copy()

            candidate_group_i_pos = [pos_list[i], itemCat_list[i]]
            candidate_group_i_neg = [neg_list[i], itemCat_list[i]]

            cons1 = behaviour_group_i + context_group_i + \
                candidate_group_i_pos + [1]
            cons2 = behaviour_group_i + context_group_i + \
                candidate_group_i_neg + [0]

            if i == len(pos_list) - 1:
                test_data.append(cons1)
                test_data.append(cons2)
            elif i == len(pos_list) - 2:
                val_data.append(cons1)
                val_data.append(cons2)
            else:
                train_data.append(cons1)
                train_data.append(cons2)

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    cols = conf_din['feature_groups']['behaviour_group'] + conf_din['feature_groups']['context_group'] + \
        conf_din['feature_groups']['candidate_group'] + ['label']
    train_df = pd.DataFrame(train_data, columns=cols)
    val_df = pd.DataFrame(val_data, columns=cols)
    test_df = pd.DataFrame(test_data, columns=cols)

    
    train, val, test = _Dataset(train_df), _Dataset(val_df), _Dataset(test_df)
    BATCH_SIZE = conf_global['batch_size']
    train_loader = DataLoader(
        train, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val,  batch_size=BATCH_SIZE,
                            collate_fn=pad_collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(
        test, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    

    col_df = train_df["behaviour_itemCat"]
    max_v = 0
    min_v = 100
    for i in col_df:
        for j in i:
            if j >max_v:
                max_v = j
            if j <min_v:
                min_v = j
    feature_itemCat = max_v - min_v + 1
    # construct feature_size_dict
    feature_size_dict = {
        'unixReviewTime': len(unixReviewTime_buckets),
        'itempos': seq_len_max + 1,
        'itemId': len(item2idx_dict) + 1,
        'itemCat': feature_itemCat,
    }
    #        'itemCat': seq_len_max + 1,

    save('../datasets/loaders.pkl', train_loader,
         val_loader, test_loader, feature_size_dict)

# create_amazon_electronic_dataset()
