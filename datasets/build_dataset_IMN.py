import random
#import pickle5 as pickle
import pickle
from utils.util import *
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch


random.seed(42)


def interval_time(min_time, ls_unixTime):
    res = []

    for time in ls_unixTime:
        unixTime = time - min_time
        right = 1
        k = 0
        while True:
            if unixTime < right:
                res.append(k)
                break
            k += 1
            right *= 2

    return res


train_set = []
test_set = []
index_dict = {
    'reviewerID': 0,
    'sequence': 1,
    'target': 2,
    'time': 3,
    'position': 4,
    'clicked': 5
}


def collate_fn(batch):
    list_seq = []
    list_len = []
    list_target = []
    list_time = []
    list_position = []

    for b in batch:
        seq = b[index_dict['sequence']]
        list_len.append(len(seq))
        list_seq.append(torch.tensor(seq))
        list_target.append(b[index_dict['target']])
        list_time.append(torch.tensor(b[index_dict['time']]))
        list_position.append(torch.tensor(b[index_dict['position']]))

    p = pad_sequence(list_position, batch_first=True)
    return pad_sequence(list_seq, batch_first=True), list_len, torch.tensor(list_target), pad_sequence(list_time, batch_first=True), pad_sequence(list_position, batch_first=True)


def create_dataset():

    pre_path = "./datasets/resources" if local_mode else "/tmp"

    with open('%s/remap_%s.pkl' % (pre_path, dataset_name), 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    max_position = max_time = 0
    for reviewerID, hist in reviews_df.groupby('reviewerID'):
        pos_list = hist['asin'].tolist()
        time_list = hist['unixReviewTime'].tolist()
        min_time = min(time_list)
        max_time = max(max_time, max(interval_time(min_time, time_list)))

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count-1)
            return neg
        neg_list = [gen_neg() for i in range(len(pos_list))]

        # pos_list : liste IDs of product that user clicked
        # neg_list : liste IDs of product that user not clicked

        # train_set : reviewerID, sequence, itemID, intervalTime, relativePosition, clicked(0/1)
        max_position = max(max_position, len(pos_list))
        for i in range(1, len(pos_list)):
            seq = pos_list[:i]
            if i != len(pos_list) - 1:
                train_set.append((reviewerID, seq, pos_list[i], interval_time(
                    min_time, time_list[:i]), [k for k in range(i)], 1))
                train_set.append((reviewerID, seq, neg_list[i], interval_time(
                    min_time, time_list[:i]), [k for k in range(i)], 0))
            else:
                label = (pos_list[i], neg_list[i])
                test_set.append((reviewerID, seq, pos_list[i], interval_time(
                    min_time, time_list[:i]), [k for k in range(i)], 1))
                test_set.append((reviewerID, seq, neg_list[i], interval_time(
                    min_time, time_list[:i]), [k for k in range(i)], 0))

    train_loader = DataLoader(train_set, shuffle=True,
                              collate_fn=collate_fn, batch_size=conf_global["batch_size"])
    test_loader = DataLoader(test_set, shuffle=True,
                             collate_fn=collate_fn, batch_size=conf_global["batch_size"])

    f = '%s/dataset_%s.pkl' % (pre_path, dataset_name)
    print('dump start, path=', f)
    #save(f, train_set)
    #save(f, test_set)
    save(f, index_dict, cate_list, user_count, item_count, cate_count, max_position,
         max_time, train_loader, test_loader)

    print('dump end')


#create_dataset()
