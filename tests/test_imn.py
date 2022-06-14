import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.util import *
from models.imn import IMN
from tqdm import tqdm
import pickle5 as pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '../datasets' if local_mode else '/tmp'

with open('%s/dataset1.pkl' % (path), 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    index_dict = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count, max_position, max_time = pickle.load(f)


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


batch_size = 128
train_loader = DataLoader(train_set, shuffle=True,
                          collate_fn=collate_fn, batch_size=batch_size)
test_loader = DataLoader(test_set, shuffle=True,
                         collate_fn=collate_fn, batch_size=batch_size)



############################# Hyper Parameters #############################
embedding_size = 258
N = 5
t = 5
lr = 1e-3
max_iter = 2

model = IMN(embedded_size=embedding_size, max_position=max_position,
            max_time=max_time, max_item=item_count, N=N, t=t)
model = model.to(device)
CE = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

for iter in tqdm(range(max_iter)):
    ls_loss = []
    for seq, length, target, time, position in train_loader:
        seq = seq.to(device)
        target = target.to(device)
        time = time.to(device)
        position = position.to(device)
        output = model(seq, target, time, position)
        #print(output, output.size())
        # dans le target, toutes les prédictions doivent être positives
        loss = CE(output, torch.ones(output.size(
            0), dtype=torch.long, device=device))
        optim.zero_grad()
        loss.backward()
        ls_loss.append(loss.item())
        optim.step()
        print(1)

    if iter % 10 == 0:
        print(iter, torch.mean(torch.tensor(ls_loss)))
