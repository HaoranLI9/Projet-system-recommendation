
from sklearn import metrics
import numpy as np
from datasets.build_dataset import _Dataset, pad_collate_fn
from datasets.build_dataset_IMN import collate_fn
from utils.util import *
import torch
from torch import optim, nn
import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.util import conf_din
from tqdm import tqdm
from pathlib import Path
from models.din import Base, DIN
from models.imn import IMN


def test(model, criterion, loader):
    loss_l = []
    auc_l = []
    for data, mask in loader:
        # seq : (seqLen, batch_size)
        behaviour_itemId, behaviour_itemCat, context_unixReviewTime, context_itempos, candidate_itemId, candidate_itemCat, y = data

        context_unixReviewTime, context_itempos, behaviour_itemId, behaviour_itemCat, candidate_itemId, candidate_itemCat, y, mask = context_unixReviewTime.to(
            device), context_itempos.to(device), behaviour_itemId.to(device), behaviour_itemCat.to(device), candidate_itemId.to(device), candidate_itemCat.to(device), y.type(torch.float32).to(device), mask.to(device)

        output = model(context_unixReviewTime, context_itempos, behaviour_itemId,
                       behaviour_itemCat, candidate_itemId, candidate_itemCat, mask)
        yhat_prob = output[:, 1]  # possibilité de label 1
        loss = criterion(yhat_prob, y)
        loss_l.append(loss.item())
        fpr, tpr, thresholds = metrics.roc_curve(
            y.detach().cpu().numpy(), yhat_prob.detach().cpu().numpy(), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        auc_l.append(auc)
    res = torch.mean(torch.tensor(loss_l)).item()
    print('test mean loss = %f' % (res))
    print('test mean auc = %f' % (np.mean(auc_l)))
    return res


def train_din(model, conf):
    maxlen = 2
    optimizer = optim.Adam(model.parameters(), lr=conf_din['runtime']['lr'])
    criterion = nn.BCELoss()
    times = []
    for iter in tqdm(range(conf_din['runtime']['epoch'])):
        starttime = datetime.datetime.now()
        train_loss = []
        for data, mask in train_loader:
            # seq : (seqLen, batch_size)
            #mask = 0
            behaviour_itemId, behaviour_itemCat, context_unixReviewTime, context_itempos, candidate_itemId, candidate_itemCat, y = data

            context_unixReviewTime, context_itempos, behaviour_itemId, behaviour_itemCat, candidate_itemId, candidate_itemCat, y, mask = context_unixReviewTime.to(
                device), context_itempos.to(device), behaviour_itemId.to(device), behaviour_itemCat.to(device), candidate_itemId.to(device), candidate_itemCat.to(device), y.type(torch.float32).to(device), mask.to(device)

            output = model(context_unixReviewTime, context_itempos, behaviour_itemId,
                           behaviour_itemCat, candidate_itemId, candidate_itemCat, mask)  # , mask)
            # print(yhat.size())
            # print(yhat)
            # print(output[:10])

            #yhat_prob = torch.max(output, dim=1).values
            # print(yhat_prob[:10])
            yhat_prob = output[:, 1]  # possibilité de label 1

            loss = criterion(yhat_prob, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        #writer.add_scalar('Loss/train', torch.mean(torch.tensor(train_loss)).item(), iter)

        endtime = datetime.datetime.now()
        time = endtime - starttime
        print('epoch', iter, 'train loss', train_loss[-1], ' time', time)
        times.append(time)
    time_mean = np.mean(times)
    print("mean time", time_mean)
    test(test_loader)


def train_imn(model, train_loader):
    creterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=conf_imn["lr"])

    for iter in tqdm(range(conf_imn['epoch'])):
        ls_loss = []
        for seq, length, target, time, position in train_loader:
            seq = seq.to(device)
            target = target.to(device)
            time = time.to(device)
            position = position.to(device)
            output = model(seq, target, time, position)
            #print(output, output.size())
            # dans le target, toutes les prédictions doivent être positives
            loss = creterion(output, torch.ones(output.size(
                0), dtype=torch.long, device=device))
            optim.zero_grad()
            loss.backward()
            ls_loss.append(loss.item())
            optim.step()

        if iter % 10 == 0:
            print(iter, torch.mean(torch.tensor(ls_loss)))


if __name__ == '__main__':
    print(conf_din)
    print(conf_imn)
    print(conf_global)
    print(conf_base)
    if conf_din['retrain'] or conf_imn['retrain']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #writer = SummaryWriter("../runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        #savepath = Path(r'../models/model_din_base.pch')
        load_path = './datasets/resources' if local_mode else '/tmp'
        # train_loader, val_loader, test_loader, feature_size_dict = load(
        #     '%s/dataset.pkl' % (load_path), 4)
        index_dict, cate_list, user_count, item_count, cate_count, max_position, max_time, train_loader, test_loader = load(
            '%s/dataset_%s.pkl' % (load_path, dataset_name), 9)
    model_names = ['BASE', 'DIN', 'IMN']
    model_dict = dict.fromkeys(model_names)
    save_path_dict = {model_name: './models/model_%s.pch' %
                      (model_name) for model_name in model_names}
    if conf_imn['retrain']:
        model_imn = IMN(embedded_size=conf_imn['emb_dim'], max_position=max_position,
                        max_time=max_time, max_item=item_count, N=conf_imn['N'], t=conf_imn['t']).to(device=device)
        train_imn(model_imn, train_loader)
        torch.save(model_imn, save_path_dict['IMN'])
    else:
        model_imn = torch.load(save_path_dict['IMN'])
        model_dict["IMN"] = model_imn

    if conf_din['retrain']:
        model_din = None
        train_din(model_din, train_loader)
        torch.save(model_din, save_path_dict['DIN'])
    else:
        model_din = torch.load(save_path_dict['DIN'])
        model_dict["IMN"] = model_din

    confs = {
        "BASE": conf_base,
        "DIN": conf_din,
        "IMN": conf_imn,
    }
