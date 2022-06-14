
import os
import sys
# add PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.din import Base
from pathlib import Path
from tqdm import tqdm
from utils.util import conf_din
from torch.utils.tensorboard import SummaryWriter
import datetime
from torch import optim, nn
import torch
from utils.util import *
from datasets.build_dataset import _Dataset, pad_collate_fn
import numpy as np
from sklearn import metrics
import datetime


def validation(loader,  iter):
    loss_l = []
    auc_l = []
    for data, mask in loader:
        # seq : (seqLen, batch_size)
        behaviour_itemId, behaviour_itemCat, context_unixReviewTime, context_itempos, candidate_itemId, candidate_itemCat, y = data

        context_unixReviewTime, context_itempos, behaviour_itemId, behaviour_itemCat, candidate_itemId, candidate_itemCat, y= context_unixReviewTime.to(device), context_itempos.to(
            device), behaviour_itemId.to(device), behaviour_itemCat.to(device), candidate_itemId.to(device), candidate_itemCat.to(device), y.type(torch.float32).to(device)
    
        output = model(context_unixReviewTime, context_itempos, behaviour_itemId,
                    behaviour_itemCat, candidate_itemId, candidate_itemCat)
        #output = torch.max(output, dim=1).values
        #yhat_prob = output[:,1]# possibilite de label 1
        
        loss = criterion(output, y.long())
        loss_l.append(loss.item())

        yhat_prob = torch.softmax(output, dim = -1)
        #print(yhat_prob[:, 1])
        fpr, tpr, thresholds = metrics.roc_curve(y.detach().cpu().numpy(), yhat_prob[:, 1].detach().cpu().numpy(), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        auc_l.append(auc)
    res = torch.mean(torch.tensor(loss_l)).item()
    print('epoch %d validation mean loss = %f' % (iter, res))
    print('epoch %d validation mean auc = %f' % (iter, np.mean(auc_l)))
    #writer.add_scalar('Loss/validation', res)
    return res

def test(loader):
    loss_l = []
    auc_l = []
    accuracys = []
    for data,mask in loader:
        # seq : (seqLen, batch_size)
        behaviour_itemId, behaviour_itemCat, context_unixReviewTime, context_itempos, candidate_itemId, candidate_itemCat, y = data

        context_unixReviewTime, context_itempos, behaviour_itemId, behaviour_itemCat, candidate_itemId, candidate_itemCat, y= context_unixReviewTime.to(
            device), context_itempos.to(device), behaviour_itemId.to(device), behaviour_itemCat.to(device), candidate_itemId.to(device), candidate_itemCat.to(device), y.type(torch.float32).to(device)
        starttime = datetime.datetime.now()
        output = model(context_unixReviewTime, context_itempos, behaviour_itemId,
                    behaviour_itemCat, candidate_itemId, candidate_itemCat)
        endtime = datetime.datetime.now()
        #yhat_prob = output[:,1]# possibilite de label 1
        #yhat_prob = output
        yhat = torch.argmax(output, dim=-1)
        loss = criterion(output, y.long())
        loss_l.append(loss.item())

        yhat_prob = torch.softmax(output, dim = -1)
        fpr, tpr, thresholds = metrics.roc_curve(y.detach().cpu().numpy(), yhat_prob[:, 1].detach().cpu().numpy(), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        auc_l.append(auc)
        accuracy = 1 - metrics.zero_one_loss(y.detach().cpu().numpy(), yhat.detach().cpu().numpy())
        accuracys.append(accuracy)
    res = torch.mean(torch.tensor(loss_l)).item()
    print('test mean loss = %f' % (res))
    print('test mean auc = %f' % (np.mean(auc_l)))
    print('test mean accuracy = %f' % (np.mean(accuracys)))
    print("inference time", endtime- starttime)
    return res

if __name__=='__main__':
    print(conf_din)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #writer = SummaryWriter("../runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    #savepath = Path(r'../models/model_din_base.pch')
    train_loader, val_loader, test_loader, feature_size_dict = load('../datasets/loaders_CD.pkl', 4)

    model = Base(feature_size_dict=feature_size_dict).to(device=device)
    maxlen = 20
    optimizer = optim.Adam(model.parameters(), lr=conf_din['runtime']['lr'], weight_decay = conf_din['runtime']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    times = []
    for iter in tqdm(range(conf_din['runtime']['epoch'])):
        starttime = datetime.datetime.now()
        train_loss = []
        #seqs = []
        for data,mask in train_loader:
            # seq : (seqLen, batch_size)
            #mask = 0
            behaviour_itemId, behaviour_itemCat, context_unixReviewTime, context_itempos, candidate_itemId, candidate_itemCat, y = data
            context_unixReviewTime, context_itempos, behaviour_itemId, behaviour_itemCat, candidate_itemId, candidate_itemCat, y= context_unixReviewTime.to(device), context_itempos.to(
                device), behaviour_itemId.to(device), behaviour_itemCat.to(device), candidate_itemId.to(device), candidate_itemCat.to(device), y.type(torch.float32).to(device)
            #print(behaviour_itemCat.shape)
            output = model(context_unixReviewTime, context_itempos, behaviour_itemId, behaviour_itemCat, candidate_itemId, candidate_itemCat)#, mask)
            # print(yhat.size())
            # print(yhat)
            #print(output[:10])
            
            #yhat_prob = torch.max(output, dim=1).values
            #print(yhat_prob[:10])
            #yhat_prob = output[:,1]# possibilité de label 1
            #yhat_prob = output
            loss = criterion(output, y.long())
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3, norm_type=0)
            optimizer.step()
            train_loss.append(loss.item())
        #print("maxseq", np.max(seqs)) 430
        #writer.add_scalar('Loss/train', torch.mean(torch.tensor(train_loss)).item(), iter)

        res = torch.mean(torch.tensor(train_loss)).item()
        endtime = datetime.datetime.now()
        time = endtime - starttime
        print('epoch', iter, 'train loss', res, ' time', time)
        times.append(time)
        validation(val_loader, iter)
    time_mean = np.mean(times)
    print("mean time", time_mean)
    test(test_loader)



